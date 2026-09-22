"""The `TrainingRunner` the worker and the CLI call to retrain on confirmed labels."""

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from loadvision.application.ports import (
    LabeledImage,
    TrainingMode,
    TrainingRequest,
    TrainingStatus,
)
from loadvision.domain.assessment import LoadPercent
from loadvision.ml.data.splits import read_folds
from loadvision.ml.models import backbones as backbones_module
from loadvision.ml.packaging.model_package import ModelPackage
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult
from loadvision.ml.pipeline.stages import (
    BuildManifest,
    BuildPackage,
    Evaluate,
    ExtractEmbeddings,
    PredictTest,
    TrainFolds,
    build_package,
)
from loadvision.ml.training import runner as runner_module
from loadvision.ml.training.gate import GateRules
from loadvision.ml.training.runner import PipelineTrainingRunner
from tests import image_files
from tests.fakes.image_storage import InMemoryImageStorage
from tests.ml.test_stages import (
    fake_checkpoint,
    fake_licenses,
    make_dataset,
    make_folds,
    toy_loader,
)
from tests.ml.test_stages import members as toy_members
from tests.model_packages import build_package as write_package
from tests.model_packages import member_data

NOW = datetime(2026, 9, 18, 9, 0, tzinfo=UTC)
CONFIG = {"members": [{"encoder": "toy_a"}, {"encoder": "toy_b"}], "version": "v2"}


class TickingClock:
    """Moves a second forward on every call, so two runs never share an id."""

    def __init__(self) -> None:
        self._calls = 0

    def __call__(self) -> datetime:
        self._calls += 1
        return NOW + timedelta(seconds=self._calls - 1)


def toy_pipeline(folds_file: Path, licenses_dir: Path, *, chain=()):
    """The six real stages, with the encoder replaced by one that needs no download."""

    def build(config, *, device="cpu"):
        return [
            BuildManifest(),
            ExtractEmbeddings(encoders=["toy_a", "toy_b"], load=toy_loader),
            TrainFolds(members=toy_members(), chain=chain, folds_file=folds_file, load=toy_loader),
            Evaluate(),
            PredictTest(members=toy_members(), chain=chain, load=toy_loader),
            BuildPackage(
                members=toy_members(),
                chain=chain,
                folds_file=folds_file,
                licenses_dir=licenses_dir,
                package_version=str(config.get("version", "v1")),
                now=NOW,
            ),
        ]

    return build


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """A base dataset, a folds file and a runner wired to toy encoders."""
    monkeypatch.chdir(tmp_path)
    data_dir = make_dataset(tmp_path / "data")
    make_folds(data_dir, tmp_path / "folds.json")
    # Relative on purpose: the manifest records the folds file by path, and an absolute one
    # would name a folder that exists only on the machine that trained. The worker therefore
    # runs with the repository root as its working directory.
    folds_file = Path("folds.json")
    licenses = fake_licenses(tmp_path)
    monkeypatch.setattr(backbones_module, "load_backbone", toy_loader)
    monkeypatch.setattr(build_package, "load_backbone", toy_loader)
    monkeypatch.setattr(build_package, "place_checkpoint", fake_checkpoint)
    monkeypatch.setattr(runner_module, "build_pipeline", toy_pipeline(folds_file, licenses))
    return data_dir, folds_file, tmp_path


def make_runner(workspace, storage: InMemoryImageStorage, **overrides) -> PipelineTrainingRunner:
    data_dir, folds_file, root = workspace
    settings = {
        "config": CONFIG,
        "data_dir": data_dir,
        "runs_dir": root / "runs",
        "cache_dir": root / "cache",
        "storage": storage,
        "clock": TickingClock(),
        "folds_file": folds_file,
        "rules": GateRules(min_new_labels=1),
    }
    return PipelineTrainingRunner(**{**settings, **overrides})


def confirm(storage: InMemoryImageStorage, index: int, load: float) -> LabeledImage:
    raw = image_files.jpeg(40 + index, 30)
    sha = hashlib.sha256(raw).hexdigest()
    storage.files[sha] = raw
    return LabeledImage(image_sha256=sha, load_pct=LoadPercent(load))


def labels(storage: InMemoryImageStorage, count: int) -> tuple[LabeledImage, ...]:
    return tuple(confirm(storage, i, float(i % 21 * 5)) for i in range(count))


@pytest.mark.ml
def test_a_run_ends_in_a_package_the_service_can_load(workspace) -> None:
    storage = InMemoryImageStorage()
    request = TrainingRequest(mode=TrainingMode.FAST, labels=labels(storage, 6))

    outcome = make_runner(workspace, storage).run(request)

    assert outcome.status is TrainingStatus.SUCCEEDED, outcome.error_detail
    assert outcome.package_path is not None
    # The gate's own check is not enough: the package must load the way the service loads it.
    package = ModelPackage.load(outcome.package_path)
    # Named after its run, not after the fixed version of the experiment config: a candidate
    # must never write over a package the service already has.
    assert package.manifest.version == outcome.run_id
    assert package.manifest.version != CONFIG["version"]
    assert outcome.gate_report["added"] == 6


@pytest.mark.ml
def test_the_first_version_passes_the_gate_and_reports_its_numbers(workspace) -> None:
    storage = InMemoryImageStorage()

    outcome = make_runner(workspace, storage).run(
        TrainingRequest(mode=TrainingMode.FAST, labels=labels(storage, 4))
    )

    assert outcome.gate_passed
    assert outcome.gate_report["reason"] == "активной версии ещё нет, кандидат становится первой"


@pytest.mark.ml
def test_a_second_run_is_compared_with_the_active_version(workspace) -> None:
    storage = InMemoryImageStorage()
    first = make_runner(workspace, storage).run(
        TrainingRequest(mode=TrainingMode.FAST, labels=labels(storage, 4))
    )
    assert first.package_path is not None

    second = make_runner(workspace, storage, active_package=first.package_path).run(
        TrainingRequest(mode=TrainingMode.FAST, labels=labels(storage, 8))
    )

    assert second.status is TrainingStatus.SUCCEEDED, second.error_detail
    # Comparable only because a stored row never changes fold.
    common = second.gate_report["common_rows"]
    assert isinstance(common, int) and common > 0
    assert second.gate_report["candidate_mae"] is not None
    assert second.gate_report["active_mae"] is not None


@pytest.mark.ml
def test_too_few_new_labels_are_refused_by_the_gate(workspace) -> None:
    storage = InMemoryImageStorage()
    first = make_runner(workspace, storage).run(
        TrainingRequest(mode=TrainingMode.FAST, labels=labels(storage, 4))
    )
    assert first.package_path is not None

    second = make_runner(
        workspace, storage, active_package=first.package_path, rules=GateRules(min_new_labels=50)
    ).run(TrainingRequest(mode=TrainingMode.FAST, labels=labels(storage, 5)))

    # A refused candidate is still a successful run: training worked, the gate did its job.
    assert second.status is TrainingStatus.SUCCEEDED
    assert not second.gate_passed
    assert second.package_path is not None


@pytest.mark.ml
def test_new_photos_get_a_fold_and_the_stored_rows_stay_put(workspace) -> None:
    folds_file = workspace[1]
    storage = InMemoryImageStorage()
    before = read_folds(folds_file)

    make_runner(workspace, storage).run(
        TrainingRequest(mode=TrainingMode.FAST, labels=labels(storage, 6))
    )

    after = read_folds(folds_file)
    assert len(after.images) == len(before.images) + 6
    assert {i: after.fold_of(i) for i in before.images} == {
        i: before.fold_of(i) for i in before.images
    }
    assert after.created_at == before.created_at


@pytest.mark.ml
def test_a_label_whose_photo_the_storage_lost_does_not_stop_the_run(workspace) -> None:
    storage = InMemoryImageStorage()
    present = labels(storage, 4)
    gone = LabeledImage(image_sha256="a" * 64, load_pct=LoadPercent(50.0))

    outcome = make_runner(workspace, storage).run(
        TrainingRequest(mode=TrainingMode.FAST, labels=(*present, gone))
    )

    assert outcome.status is TrainingStatus.SUCCEEDED, outcome.error_detail
    assert outcome.gate_report["missing"] == 1
    assert outcome.gate_report["added"] == 4


@pytest.mark.ml
def test_retraining_on_the_base_dataset_alone_is_allowed(workspace) -> None:
    outcome = make_runner(workspace, InMemoryImageStorage()).run(
        TrainingRequest(mode=TrainingMode.FAST)
    )

    assert outcome.status is TrainingStatus.SUCCEEDED, outcome.error_detail
    assert outcome.gate_report["added"] == 0


@pytest.mark.ml
def test_full_mode_is_refused_because_it_is_not_built(workspace) -> None:
    outcome = make_runner(workspace, InMemoryImageStorage()).run(
        TrainingRequest(mode=TrainingMode.FULL)
    )

    assert outcome.status is TrainingStatus.FAILED
    assert outcome.package_path is None
    assert "fine-tunes the encoder" in (outcome.error_detail or "")


@pytest.mark.ml
def test_a_failing_stage_comes_back_in_the_outcome_not_as_an_exception(
    workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Breaking:
        name = "evaluate"
        version = "1"

        def inputs(self, ctx: RunContext) -> dict[str, str]:
            return {}

        def run(self, ctx: RunContext) -> StageResult:
            return StageResult()

        def check(self, ctx: RunContext, result: StageResult) -> None:
            raise StageCheckError("oof misses 5 rows")

    def build(config, *, device="cpu"):
        return [BuildManifest(), Breaking()]

    monkeypatch.setattr(runner_module, "build_pipeline", build)
    storage = InMemoryImageStorage()

    outcome = make_runner(workspace, storage).run(
        TrainingRequest(mode=TrainingMode.FAST, labels=labels(storage, 4))
    )

    # The worker records this in the job and keeps serving the model it already has.
    assert outcome.status is TrainingStatus.FAILED
    assert outcome.error_code is not None
    assert "oof misses 5 rows" in (outcome.error_detail or "")


@pytest.mark.ml
def test_the_base_dataset_is_never_written_to(workspace) -> None:
    data_dir = workspace[0]
    storage = InMemoryImageStorage()
    before = sorted(p.name for p in (data_dir / "train" / "images").iterdir())
    csv_before = (data_dir / "train" / "train.csv").read_text(encoding="utf-8")

    make_runner(workspace, storage).run(
        TrainingRequest(mode=TrainingMode.FAST, labels=labels(storage, 6))
    )

    assert sorted(p.name for p in (data_dir / "train" / "images").iterdir()) == before
    assert (data_dir / "train" / "train.csv").read_text(encoding="utf-8") == csv_before


@pytest.mark.ml
def test_two_runs_get_their_own_folders(workspace) -> None:
    storage = InMemoryImageStorage()
    runner = make_runner(workspace, storage)
    request = TrainingRequest(mode=TrainingMode.FAST, labels=labels(storage, 4))

    first = runner.run(request)
    second = runner.run(request)

    assert first.run_id != second.run_id
    assert first.package_path != second.package_path


@pytest.mark.ml
def test_a_package_shaped_like_the_served_one_has_its_heads_refitted(
    workspace, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The candidate must come from the family that is served, or the gate compares nothing."""
    _data_dir, folds_file, root = workspace
    folds = read_folds(folds_file).n_folds
    active = write_package(
        root / "active",
        {f"m{index}/head.npz": b"head" for index in range(folds)}
        | {f"w{index}.safetensors": f"weights {index}".encode() for index in range(folds)},
        members=[
            member_data(
                backbone="dinov2_vitb14",
                runtime="torch",
                head=f"m{index}/head.npz",
                weights=f"w{index}.safetensors",
                weight=1.0 / folds,
            )
            for index in range(folds)
        ],
    )
    called: list[str] = []
    monkeypatch.setattr(
        runner_module, "refit_heads", lambda **kwargs: called.append(kwargs["version"])
    )
    monkeypatch.setattr(
        runner_module.PipelineTrainingRunner,
        "_pipeline",
        lambda *args, **kwargs: pytest.fail("the pipeline must not run for a refittable package"),
    )
    storage = InMemoryImageStorage()

    make_runner(workspace, storage, active_package=active.root).run(
        TrainingRequest(mode=TrainingMode.FAST, labels=labels(storage, 2))
    )

    assert called
