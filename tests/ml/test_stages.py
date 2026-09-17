import csv
import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from loadvision.ml.data.manifest import load_split
from loadvision.ml.data.splits import build_folds, write_folds
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.models import backbones as backbones_module
from loadvision.ml.models.backbones import Backbone
from loadvision.ml.packaging.model_package import ModelPackage
from loadvision.ml.pipeline.runner import PipelineRunner, RunStatus
from loadvision.ml.pipeline.stage import RunContext, StageCheckError, StageResult
from loadvision.ml.pipeline.stages import (
    BuildManifest,
    BuildPackage,
    Evaluate,
    ExtractEmbeddings,
    PredictTest,
    TrainFolds,
    build_package,
    build_pipeline,
)
from loadvision.ml.pipeline.stages.train_folds import Member
from loadvision.ml.registry import BackboneSpec
from loadvision.ml.training.embedding_cache import read_image

DIM = 6
SPEC = BackboneSpec(
    timm_name="fake", patch=8, height=32, width=24, license="Toy-1.0", origin="a test"
)
PARAMS = {
    "height": 32,
    "width": 24,
    "mean_r": 0.5,
    "mean_g": 0.5,
    "mean_b": 0.5,
    "std_r": 0.5,
    "std_g": 0.5,
    "std_b": 0.5,
}
NOW = datetime(2026, 9, 17, 10, 0, 0, tzinfo=UTC)


class ToyEncoder(torch.nn.Module):
    """Stand-in encoder with the two halves timm models have, so pooling can be shared.

    Its features are the mean of every channel and its square, enough for a head to learn.
    """

    def forward_features(self, batch: torch.Tensor) -> torch.Tensor:
        means = batch.mean(dim=(2, 3))
        tokens = torch.cat([means, means**2], dim=1)
        return tokens[:, None, :].repeat(1, 12, 1)

    def forward_head(self, tokens: torch.Tensor) -> torch.Tensor:
        return tokens.mean(dim=1)

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        return self.forward_head(self.forward_features(batch))


def fake_licenses(root: Path) -> Path:
    """Writes the licence text the toy spec names, so a package can carry it."""
    folder = root / "licenses"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "Toy-1.0.txt").write_text("Toy licence, for tests only.\n", encoding="utf-8")
    return folder


def fake_checkpoint(root: Path, timm_name: str) -> Path:
    """Stands in for the pretrained weights, so a test writes bytes instead of 340 MB."""
    target = root / "weights" / f"{timm_name}.safetensors"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"weights of " + timm_name.encode())
    return target


def toy_loader(name: str, _device: str) -> Backbone:
    return Backbone(
        name=name,
        module=ToyEncoder(),
        preprocessing=SPEC.preprocessing,
        params=PARAMS,
        dim=DIM,
        spec=SPEC,
    )


def make_dataset(root: Path, train_photos: int = 84, test_photos: int = 12) -> Path:
    """Writes a dataset where the load is readable from how red a photo is."""
    rng = np.random.default_rng(0)
    for split, count in (("train", train_photos), ("test", test_photos)):
        images = root / split / "images"
        images.mkdir(parents=True)
        rows: list[tuple[str, float]] = []
        for index in range(count):
            label = float((index % 21) * 5)
            red = round(label * 2.55)
            noise = int(rng.integers(-5, 6))
            image_id = f"img_{split}_{index:03d}"
            Image.new("RGB", (40, 30), (max(0, min(255, red + noise)), 10, 10)).save(
                images / f"{image_id}.jpg"
            )
            rows.append((image_id, label))
        if split == "train":
            lines = ["image_id,load_pct"] + [f"{name},{value:g}" for name, value in rows]
            (root / split / "train.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
            groups = ["image_id,group_id"] + [
                f"{name},grp_{index:03d}" for index, (name, _) in enumerate(rows)
            ]
            (root / split / "train_groups.csv").write_text(
                "\n".join(groups) + "\n", encoding="utf-8"
            )
        else:
            lines = ["image_id"] + [name for name, _ in rows]
            (root / split / "test.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return root


def make_folds(data_dir: Path, path: Path) -> Path:
    samples = load_split(data_dir, "train")
    folds = build_folds(
        samples,
        now=NOW,
        dedup={"encoder": "fake", "metric": "cosine", "threshold": 0.98},
        n_folds=3,
    )
    write_folds(path, folds)
    return path


@pytest.fixture
def workspace(tmp_path: Path) -> tuple[RunContext, Path]:
    data_dir = make_dataset(tmp_path / "data")
    folds_file = make_folds(data_dir, tmp_path / "folds.json")
    ctx = RunContext(
        run_id="run_test_000001_abcdef",
        runs_dir=tmp_path / "runs",
        cache_dir=tmp_path / "cache",
        data_dir=data_dir,
        config={},
        seed=0,
    )
    ctx.run_dir.mkdir(parents=True)
    return ctx, folds_file


def members() -> Sequence[Member]:
    return [
        Member(encoder="toy_a", params={"alpha": 1.0}),
        Member(encoder="toy_b", params={"alpha": 10.0}),
    ]


def run_upto(ctx: RunContext, folds_file: Path, chain: Sequence[tuple[str, dict]] = ()) -> None:
    BuildManifest().run(ctx)
    ExtractEmbeddings(encoders=["toy_a", "toy_b"], load=toy_loader).run(ctx)
    TrainFolds(members=members(), chain=chain, folds_file=folds_file, load=toy_loader).run(ctx)


@pytest.mark.ml
def test_build_manifest_writes_every_photo(workspace: tuple[RunContext, Path]) -> None:
    ctx, _ = workspace
    stage = BuildManifest()

    result = stage.run(ctx)
    stage.check(ctx, result)

    assert result.counts["train"] == 84
    assert result.counts["test"] == 12
    with (ctx.run_dir / "manifest.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 96
    assert {row["split"] for row in rows} == {"train", "test"}
    # The slice columns are what the report groups photos by; they must be filled for every
    # photo, labelled or not.
    assert all(row["width"] and row["height"] and row["aspect"] for row in rows)
    assert all(0.0 <= float(row["brightness"]) <= 255.0 for row in rows)
    assert {row["aspect"] for row in rows} == {f"{40 / 30:.4f}"}


@pytest.mark.ml
def test_build_manifest_check_catches_a_lost_file(workspace: tuple[RunContext, Path]) -> None:
    ctx, _ = workspace
    stage = BuildManifest()
    result = stage.run(ctx)
    (ctx.run_dir / "manifest.csv").unlink()

    with pytest.raises(StageCheckError, match="manifest was not written"):
        stage.check(ctx, result)


@pytest.mark.ml
def test_extract_embeddings_covers_both_splits(workspace: tuple[RunContext, Path]) -> None:
    ctx, _ = workspace
    stage = ExtractEmbeddings(encoders=["toy_a"], load=toy_loader)

    result = stage.run(ctx)
    stage.check(ctx, result)

    assert result.counts["photos"] == 96
    assert result.counts["dim_toy_a"] == DIM
    assert (ctx.cache_dir / "embeddings" / "toy_a.npz").is_file()


@pytest.mark.ml
def test_extract_embeddings_check_catches_a_missing_cache(
    workspace: tuple[RunContext, Path],
) -> None:
    ctx, _ = workspace
    stage = ExtractEmbeddings(encoders=["toy_a"], load=toy_loader)
    result = stage.run(ctx)
    (ctx.cache_dir / "embeddings" / "toy_a.npz").unlink()

    with pytest.raises(StageCheckError, match="has no cache file"):
        stage.check(ctx, result)


@pytest.mark.ml
def test_train_folds_predicts_every_photo_once(workspace: tuple[RunContext, Path]) -> None:
    ctx, folds_file = workspace
    BuildManifest().run(ctx)
    ExtractEmbeddings(encoders=["toy_a", "toy_b"], load=toy_loader).run(ctx)
    stage = TrainFolds(members=members(), folds_file=folds_file, load=toy_loader)

    result = stage.run(ctx)
    stage.check(ctx, result)

    with (ctx.run_dir / "oof.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 84
    assert len({row["image_id"] for row in rows}) == 84
    assert {int(row["fold"]) for row in rows} == {0, 1, 2}


@pytest.mark.ml
def test_train_folds_check_catches_predictions_out_of_range(
    workspace: tuple[RunContext, Path],
) -> None:
    ctx, folds_file = workspace
    run_upto(ctx, folds_file)
    stage = TrainFolds(members=members(), folds_file=folds_file, load=toy_loader)
    result = StageResult(artifacts={"oof": ctx.run_dir / "oof.csv"})
    path = ctx.run_dir / "oof.csv"
    broken = path.read_text(encoding="utf-8").replace("\n", "\n", 1).splitlines()
    broken[1] = broken[1].rsplit(",", 1)[0] + ",500.0"
    path.write_text("\n".join(broken) + "\n", encoding="utf-8")

    with pytest.raises(StageCheckError, match=r"outside \[0, 100\]"):
        stage.check(ctx, result)


@pytest.mark.ml
def test_postprocessing_runs_inside_train_folds(workspace: tuple[RunContext, Path]) -> None:
    ctx, folds_file = workspace
    run_upto(ctx, folds_file, chain=[("round_to", {"step": 5.0})])

    with (ctx.run_dir / "oof.csv").open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    values = np.array([float(row["predicted"]) for row in rows])
    assert np.all(np.isclose(values % 5.0, 0.0))


@pytest.mark.ml
def test_evaluate_writes_metrics(workspace: tuple[RunContext, Path]) -> None:
    ctx, folds_file = workspace
    run_upto(ctx, folds_file)
    stage = Evaluate()

    result = stage.run(ctx)
    stage.check(ctx, result)

    metrics = json.loads((ctx.run_dir / "metrics.json").read_text(encoding="utf-8"))
    assert {"mae", "within_10", "p95_error", "mae_fold_spread"} <= set(metrics)
    assert (ctx.run_dir / "worst_oof.csv").is_file()
    # The toy encoder sees the colour the label was drawn from, so it must beat guessing.
    assert metrics["mae"] < 20.0


@pytest.mark.ml
def test_evaluate_needs_training_first(workspace: tuple[RunContext, Path]) -> None:
    ctx, _ = workspace

    with pytest.raises(FileNotFoundError, match="train_folds did not run"):
        Evaluate().run(ctx)


@pytest.mark.ml
def test_predict_test_writes_a_valid_submission(workspace: tuple[RunContext, Path]) -> None:
    ctx, folds_file = workspace
    run_upto(ctx, folds_file)
    stage = PredictTest(members=members(), load=toy_loader)

    result = stage.run(ctx)
    stage.check(ctx, result)

    with (ctx.run_dir / "submission.csv").open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        assert reader.fieldnames == ["image_id", "load_pct"]
        rows = list(reader)
    assert len(rows) == 12
    values = [float(row["load_pct"]) for row in rows]
    assert min(values) >= 0.0
    assert max(values) <= 100.0


@pytest.mark.ml
def test_predict_test_check_catches_wrong_ids(workspace: tuple[RunContext, Path]) -> None:
    ctx, folds_file = workspace
    run_upto(ctx, folds_file)
    stage = PredictTest(members=members(), load=toy_loader)
    result = stage.run(ctx)
    path = ctx.run_dir / "submission.csv"
    path.write_text("image_id,load_pct\nimg_ghost,50.0\n", encoding="utf-8")

    with pytest.raises(StageCheckError, match="do not match"):
        stage.check(ctx, result)


@pytest.mark.ml
def test_the_whole_pipeline_runs_and_records_itself(workspace: tuple[RunContext, Path]) -> None:
    ctx, folds_file = workspace
    stages = [
        BuildManifest(),
        ExtractEmbeddings(encoders=["toy_a", "toy_b"], load=toy_loader),
        TrainFolds(
            members=members(),
            chain=[("isotonic", {}), ("round_to", {"step": 5.0})],
            folds_file=folds_file,
            load=toy_loader,
        ),
        Evaluate(),
        PredictTest(members=members(), chain=[("isotonic", {})], load=toy_loader),
    ]
    fresh = RunContext(
        run_id="run_test_000002_abcdef",
        runs_dir=ctx.runs_dir,
        cache_dir=ctx.cache_dir,
        data_dir=ctx.data_dir,
        config={},
        seed=0,
    )

    summary = PipelineRunner(clock=lambda: NOW).run(stages, fresh)

    assert summary.status is RunStatus.SUCCEEDED
    assert [stage.name for stage in summary.stages] == [
        "build_manifest",
        "extract_embeddings",
        "train_folds",
        "evaluate",
        "predict_test",
    ]
    assert (fresh.run_dir / "summary.json").is_file()
    assert (fresh.run_dir / "submission.csv").is_file()


@pytest.mark.ml
def test_a_failing_stage_stops_the_run(workspace: tuple[RunContext, Path]) -> None:
    ctx, _ = workspace
    fresh = RunContext(
        run_id="run_test_000003_abcdef",
        runs_dir=ctx.runs_dir,
        cache_dir=ctx.cache_dir,
        data_dir=ctx.data_dir,
        config={},
        seed=0,
    )

    # evaluate before train_folds: the run must stop and say so, not crash the process.
    summary = PipelineRunner(clock=lambda: NOW).run([Evaluate(), BuildManifest()], fresh)

    assert summary.status is RunStatus.FAILED
    assert summary.failed_stage is not None
    assert summary.failed_stage.name == "evaluate"
    assert summary.stages[1].status.value == "not_run"


@pytest.mark.ml
def test_build_pipeline_reads_a_config() -> None:
    stages = build_pipeline(
        {
            "members": [
                {"encoder": "dinov3_vitb16", "head": "ridge", "params": {"alpha": 200}},
                {"encoder": "pe_spatial_vitb16", "params": {"alpha": 50}},
            ],
            "postprocess": [{"name": "round_to", "params": {"step": 5}}, "clip"],
        }
    )

    assert [stage.name for stage in stages] == [
        "build_manifest",
        "extract_embeddings",
        "train_folds",
        "evaluate",
        "predict_test",
        "build_package",
    ]


@pytest.mark.ml
@pytest.mark.parametrize(
    "config",
    [
        {},
        {"members": []},
        {"members": [{"head": "ridge"}]},
        {"members": [{"encoder": "x", "params": {"alpha": "many"}}]},
    ],
)
def test_build_pipeline_rejects_a_broken_config(config: dict) -> None:
    with pytest.raises(ValueError, match="member"):
        build_pipeline(config)


@pytest.mark.ml
def test_build_pipeline_rejects_a_broken_postprocess_entry() -> None:
    with pytest.raises(ValueError, match="needs 'name'"):
        build_pipeline({"members": [{"encoder": "x"}], "postprocess": [{"step": "round_to"}]})


@pytest.mark.ml
def test_evaluate_reports_every_scenario_it_finds(workspace: tuple[RunContext, Path]) -> None:
    ctx, folds_file = workspace
    run_upto(ctx, folds_file)

    result = Evaluate().run(ctx)
    Evaluate().check(ctx, result)

    cases = json.loads((ctx.run_dir / "cases.json").read_text(encoding="utf-8"))
    # The synthetic dataset walks the whole range, so the load scenarios are all present.
    assert {"empty", "low", "middle", "high", "full"} <= set(cases)
    for name, case in cases.items():
        assert case["photos"] > 0, name
        assert case["description"], name
        assert 0.0 <= case["within_10"] <= 1.0, name
        assert len(case["worst"]) <= Evaluate().examples_kept, name
    assert any(key.startswith("mae_case_") for key in result.metrics)


@pytest.mark.ml
def test_evaluate_needs_the_manifest_for_scenarios(workspace: tuple[RunContext, Path]) -> None:
    ctx, folds_file = workspace
    run_upto(ctx, folds_file)
    (ctx.run_dir / "manifest.csv").unlink()

    with pytest.raises(FileNotFoundError, match="build_manifest did not run"):
        Evaluate().run(ctx)


@pytest.mark.ml
def test_build_package_writes_a_package_the_service_can_load(
    workspace: tuple[RunContext, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, folds_file = workspace
    run_upto(ctx, folds_file, chain=[("round_to", {"step": 5.0})])
    monkeypatch.setattr(build_package, "load_backbone", toy_loader)
    monkeypatch.setattr(build_package, "_place_checkpoint", fake_checkpoint)
    # The manifest records the folds by their repository path, never an absolute one.
    monkeypatch.chdir(folds_file.parent)
    stage = BuildPackage(
        members=members(),
        chain=[("round_to", {"step": 5.0})],
        folds_file=Path(folds_file.name),
        licenses_dir=fake_licenses(ctx.runs_dir),
    )

    result = stage.run(ctx)

    manifest = json.loads((ctx.run_dir / "package" / "manifest.json").read_text(encoding="utf-8"))
    assert len(manifest["members"]) == len(members())
    assert [step["name"] for step in manifest["postprocess"]] == ["round_to"]
    assert manifest["cv"]["mae"] > 0
    assert manifest["folds"]["file"] == folds_file.name
    assert result.counts["members"] == len(members())
    # Every file the manifest names is listed with its hash, so the loader can verify it.
    assert set(manifest["files"]) >= {"m0/head.npz", "m1/head.npz"}


@pytest.mark.ml
def test_members_sharing_a_checkpoint_share_its_file(
    workspace: tuple[RunContext, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, folds_file = workspace
    run_upto(ctx, folds_file)
    monkeypatch.setattr(build_package, "load_backbone", toy_loader)
    monkeypatch.setattr(build_package, "_place_checkpoint", fake_checkpoint)
    monkeypatch.chdir(folds_file.parent)

    BuildPackage(
        members=members(),
        folds_file=Path(folds_file.name),
        licenses_dir=fake_licenses(ctx.runs_dir),
    ).run(ctx)

    manifest = json.loads((ctx.run_dir / "package" / "manifest.json").read_text(encoding="utf-8"))
    # Both toy members name the same checkpoint, so the package carries one copy of it.
    assert len({member["weights"] for member in manifest["members"]}) == 1


@pytest.mark.ml
def test_the_package_answers_exactly_what_the_submission_says(
    workspace: tuple[RunContext, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The service and the platform must see one model, not two that share a config.

    `predict_test` and `build_package` fit the same heads and the same chain by two separate
    code paths, so this is the only check that keeps them from drifting apart.
    """
    ctx, folds_file = workspace
    chain = [("median_bins", {"bins": 4.0}), ("round_to", {"step": 5.0})]
    run_upto(ctx, folds_file, chain)
    PredictTest(members=members(), chain=chain, load=toy_loader).run(ctx)
    monkeypatch.setattr(backbones_module, "load_backbone", toy_loader)
    monkeypatch.setattr(build_package, "load_backbone", toy_loader)
    monkeypatch.setattr(build_package, "_place_checkpoint", fake_checkpoint)
    monkeypatch.chdir(folds_file.parent)
    stage = BuildPackage(
        members=members(),
        chain=chain,
        folds_file=Path(folds_file.name),
        licenses_dir=fake_licenses(ctx.runs_dir),
    )

    result = stage.run(ctx)
    stage.check(ctx, result)

    predictor = PackagePredictor(ModelPackage.load(ctx.run_dir / "package"))
    test_photos = load_split(ctx.data_dir, "test")
    answered = [
        prediction.load_pct.value
        for prediction in predictor.predict([read_image(sample.path) for sample in test_photos])
    ]
    with (ctx.run_dir / "submission.csv").open(newline="", encoding="utf-8") as file:
        submitted = {row["image_id"]: float(row["load_pct"]) for row in csv.DictReader(file)}
    assert answered == pytest.approx(
        [submitted[sample.image_id] for sample in test_photos], abs=0.05
    )


@pytest.mark.ml
def test_the_package_carries_the_licence_of_every_checkpoint(
    workspace: tuple[RunContext, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    ctx, folds_file = workspace
    run_upto(ctx, folds_file)
    monkeypatch.setattr(build_package, "load_backbone", toy_loader)
    monkeypatch.setattr(build_package, "_place_checkpoint", fake_checkpoint)
    monkeypatch.chdir(folds_file.parent)

    result = BuildPackage(
        members=members(),
        folds_file=Path(folds_file.name),
        licenses_dir=fake_licenses(ctx.runs_dir),
    ).run(ctx)

    root = ctx.run_dir / "package"
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["licenses"] == ["licenses/NOTICE.md", "licenses/Toy-1.0.txt"]
    # Listed in `files` too, so the loader verifies the texts like every other package file.
    assert set(manifest["licenses"]) <= set(manifest["files"])
    notice = (root / "licenses" / "NOTICE.md").read_text(encoding="utf-8")
    assert "fake" in notice and "Toy-1.0" in notice and "a test" in notice
    assert result.counts["licenses"] == 2


@pytest.mark.ml
def test_a_checkpoint_without_its_licence_text_stops_the_build(
    workspace: tuple[RunContext, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Shipping weights whose terms we cannot name is worse than failing the run."""
    ctx, folds_file = workspace
    run_upto(ctx, folds_file)
    monkeypatch.setattr(build_package, "load_backbone", toy_loader)
    monkeypatch.setattr(build_package, "_place_checkpoint", fake_checkpoint)
    monkeypatch.chdir(folds_file.parent)

    with pytest.raises(FileNotFoundError, match=r"Toy-1\.0"):
        BuildPackage(
            members=members(),
            folds_file=Path(folds_file.name),
            licenses_dir=ctx.runs_dir / "empty",
        ).run(ctx)


@pytest.mark.ml
def test_build_package_needs_training_first(workspace: tuple[RunContext, Path]) -> None:
    ctx, folds_file = workspace
    BuildManifest().run(ctx)
    ExtractEmbeddings(encoders=["toy_a", "toy_b"], load=toy_loader).run(ctx)

    with pytest.raises(FileNotFoundError, match="train_folds did not run"):
        BuildPackage(
            members=members(), chain=[("round_to", {"step": 5.0})], folds_file=folds_file
        ).run(ctx)
