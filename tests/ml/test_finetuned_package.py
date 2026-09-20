import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from loadvision.ml.data.manifest import load_split
from loadvision.ml.packaging.finetuned_package import (
    build_finetuned_package,
    build_folds_package,
)
from loadvision.ml.packaging.model_package import ModelPackage, file_sha256
from loadvision.ml.pipeline.stages import DEFAULT_CHAIN
from loadvision.ml.pipeline.stages.train_folds import OOF_FILE, read_oof
from tests.ml.test_stages import fake_licenses, toy_loader
from tests.unit.test_cli_finetune import (
    PHOTOS_ENOUGH_TO_CALIBRATE,
    arguments,
    make_workspace,
    run,
    run_folder,
)


@pytest.fixture(autouse=True)
def toy_encoder(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("loadvision.ml.packaging.finetuned_package.load_backbone", toy_loader)


@pytest.fixture(scope="module")
def runs(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Path]:
    """An out-of-fold run and a full run of the same settings, as a GPU session leaves them."""
    root = tmp_path_factory.mktemp("runs")
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr("loadvision.ml.training.finetune.load_backbone", toy_loader)
        workspace = make_workspace(root, PHOTOS_ENOUGH_TO_CALIBRATE)
        run(*arguments(workspace, "--no-weights"))
        oof = run_folder(workspace["runs"])
        workspace["runs"] = root / "fold_runs"
        run(*arguments(workspace))
        with_weights = run_folder(workspace["runs"])
        workspace["runs"] = root / "full_runs"
        run(*arguments(workspace, "--full"))
    return {
        **workspace,
        "oof": oof,
        "oof_weights": with_weights,
        "full": run_folder(workspace["runs"]),
        "root": root,
    }


def build(runs: dict[str, Path], out: Path, oof_run: Path | None = None) -> ModelPackage:
    oof_file = (oof_run or runs["oof"]) / OOF_FILE
    return build_finetuned_package(
        runs["full"],
        out,
        encoder="toy",
        train=load_split(runs["data"], "train"),
        oof=read_oof(oof_file),
        oof_file=oof_file,
        chain=[(name, dict(params)) for name, params in DEFAULT_CHAIN],
        # The manifest records the folds file relative to the checkout, as `configs/...` is.
        folds_file=runs["folds"].relative_to(runs["root"]),
        licenses_dir=fake_licenses(out.parent),
        version="v2",
        git_commit="0" * 40,
        data_hash="0" * 64,
        created_at=datetime(2026, 9, 19, tzinfo=UTC),
        device="cpu",
    )


@pytest.mark.ml
def test_the_package_carries_the_fine_tuned_model_and_its_out_of_fold_record(
    runs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(runs["root"])
    package = build(runs, tmp_path / "package")

    manifest = ModelPackage.load(package.root).manifest
    [member] = manifest.members
    assert member.weights is not None
    # The encoder the service runs is the fine-tuned one, byte for byte.
    assert file_sha256(package.path(member.weights)) == file_sha256(
        runs["full"] / "encoder.safetensors"
    )
    assert file_sha256(package.path(member.head)) == file_sha256(runs["full"] / "head.npz")
    assert manifest.cv is not None
    assert manifest.cv.mae == pytest.approx(
        json.loads((runs["oof"] / "metrics.json").read_text())["mae"]
    )
    assert [step.name for step in manifest.postprocess] == [name for name, _ in DEFAULT_CHAIN]
    assert manifest.ood is not None
    notice = (package.root / "licenses" / "NOTICE.md").read_text(encoding="utf-8")
    assert "веса дообучены" in notice


@pytest.mark.ml
def test_an_out_of_fold_run_on_other_photos_is_refused(
    runs: dict[str, Path], tmp_path: Path
) -> None:
    partial = tmp_path / "partial"
    partial.mkdir()
    lines = (runs["oof"] / OOF_FILE).read_text(encoding="utf-8").splitlines()
    (partial / OOF_FILE).write_text("\n".join(lines[:-1]) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="exactly the training photos"):
        build(runs, tmp_path / "package", oof_run=partial)
    assert not (tmp_path / "package").exists()


@pytest.mark.ml
def test_the_fold_models_of_a_run_become_one_package_of_equal_members(
    runs: dict[str, Path], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(runs["root"])
    oof_file = runs["oof_weights"] / OOF_FILE
    package = build_folds_package(
        runs["oof_weights"],
        tmp_path / "package",
        encoder="toy",
        train=load_split(runs["data"], "train"),
        oof=read_oof(oof_file),
        oof_file=oof_file,
        chain=[(name, dict(params)) for name, params in DEFAULT_CHAIN],
        folds_file=runs["folds"].relative_to(runs["root"]),
        licenses_dir=fake_licenses(tmp_path),
        version="v3",
        git_commit="0" * 40,
        data_hash="0" * 64,
        created_at=datetime(2026, 9, 20, tzinfo=UTC),
        device="cpu",
    )

    members = ModelPackage.load(package.root).manifest.members
    # One member per fold, averaged in equal shares, each with its own weights.
    assert len(members) == 3
    assert [member.weight for member in members] == pytest.approx([1 / 3] * 3)
    assert len({member.weights for member in members}) == 3
    for fold, member in enumerate(members):
        assert member.weights is not None
        assert file_sha256(package.path(member.weights)) == file_sha256(
            runs["oof_weights"] / "folds" / f"fold_{fold}" / "encoder.safetensors"
        )


@pytest.mark.ml
def test_a_run_that_kept_no_fold_models_is_refused(runs: dict[str, Path], tmp_path: Path) -> None:
    oof_file = runs["oof"] / OOF_FILE
    with pytest.raises(FileNotFoundError, match="no fold models"):
        build_folds_package(
            runs["oof"],
            tmp_path / "package",
            encoder="toy",
            train=load_split(runs["data"], "train"),
            oof=read_oof(oof_file),
            oof_file=oof_file,
            chain=[],
            folds_file=runs["folds"],
            licenses_dir=fake_licenses(tmp_path),
            version="v3",
            git_commit="0" * 40,
            data_hash="0" * 64,
            created_at=datetime(2026, 9, 20, tzinfo=UTC),
            device="cpu",
        )
