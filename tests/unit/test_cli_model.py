import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import yaml

from loadvision.cli.main import main
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.model_package import ModelPackage
from loadvision.settings import CONFIG_PATH_ENV, DEFAULT_CONFIG_PATH
from tests.ml.test_dataset_manifest import write_split

FEATURES = 233
INITIAL_VERSION = "baseline-v0"


@dataclass(frozen=True)
class Workspace:
    models_dir: Path
    baseline_dir: Path
    data_dir: Path

    def build(self, *extra: str) -> int:
        return main(
            [
                "model",
                "build-baseline",
                "--baseline-dir",
                str(self.baseline_dir),
                "--data",
                str(self.data_dir),
                *extra,
            ]
        )


@pytest.fixture
def workspace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Workspace:
    settings = yaml.safe_load(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    settings["storage"] = {
        "images_dir": str(tmp_path / "images"),
        "models_dir": str(tmp_path / "models"),
    }
    settings["model"] = {"version": INITIAL_VERSION}
    config = tmp_path / "app.yaml"
    config.write_text(yaml.safe_dump(settings), encoding="utf-8")
    monkeypatch.setenv(CONFIG_PATH_ENV, str(config))

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    np.savez(
        baseline_dir / "model.npz",
        mean=np.zeros(FEATURES),
        scale=np.ones(FEATURES),
        weights=np.zeros(FEATURES),
        intercept=np.asarray(55.0),
    )

    data_dir = tmp_path / "data"
    write_split(data_dir, "train", [("img_a", "10.0")], groups=[("img_a", "grp_1")])
    write_split(data_dir, "test", [("img_b", "")])
    return Workspace(models_dir=tmp_path / "models", baseline_dir=baseline_dir, data_dir=data_dir)


def head_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()


@pytest.mark.unit
def test_baseline_package_is_built_for_the_initial_version_and_the_service_loads_it(
    workspace: Workspace,
) -> None:
    assert workspace.build() == 0

    package = ModelPackage.load(workspace.models_dir / INITIAL_VERSION)
    assert package.version.value == INITIAL_VERSION
    assert package.manifest.git_commit == head_commit()
    pixels = np.zeros((8, 6, 3), dtype=np.uint8)
    [prediction] = PackagePredictor(package).predict(
        [PreparedImage(b"photo", MediaType.JPEG, pixels)]
    )
    assert prediction.load_pct.value == pytest.approx(55.0)


@pytest.mark.unit
def test_dataset_hash_changes_with_the_labels_and_only_with_them(workspace: Workspace) -> None:
    assert workspace.build("--version", "v1") == 0
    assert workspace.build("--version", "v2") == 0
    train_csv = workspace.data_dir / "train" / "train.csv"
    train_csv.write_text("image_id,load_pct\nimg_a,20.0\n", encoding="utf-8")
    assert workspace.build("--version", "v3") == 0

    hashes = [
        ModelPackage.load(workspace.models_dir / version).manifest.data_hash
        for version in ("v1", "v2", "v3")
    ]
    assert hashes[0] == hashes[1] != hashes[2]


@pytest.mark.unit
def test_existing_package_folder_is_never_touched(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    existing = workspace.models_dir / INITIAL_VERSION
    existing.mkdir(parents=True)

    assert workspace.build() == 1

    assert list(existing.iterdir()) == []
    assert "already exists" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize("version", ["../outside", ".hidden", ""])
def test_invalid_version_is_refused(
    workspace: Workspace, caplog: pytest.LogCaptureFixture, version: str
) -> None:
    assert workspace.build("--version", version) == 1
    assert not workspace.models_dir.exists()
    assert "invalid version" in caplog.text


@pytest.mark.unit
def test_missing_weights_leave_no_package_folder(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    (workspace.baseline_dir / "model.npz").unlink()

    assert workspace.build() == 1

    assert not (workspace.models_dir / INITIAL_VERSION).exists()
    assert "baseline weights not found" in caplog.text


@pytest.mark.unit
def test_inconsistent_dataset_leaves_no_package_folder(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    (workspace.data_dir / "train" / "images" / "img_extra.jpg").write_bytes(b"extra")

    assert workspace.build() == 1

    assert not workspace.models_dir.exists()
    assert "missing from the csv" in caplog.text


@pytest.mark.unit
def test_missing_dataset_split_is_reported(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    shutil.rmtree(workspace.data_dir / "test")

    assert workspace.build() == 1

    assert not workspace.models_dir.exists()
    assert "images folder not found" in caplog.text


@pytest.mark.unit
def test_build_outside_a_git_checkout_is_refused(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("GIT_DIR", str(workspace.data_dir / "no-repository"))

    assert workspace.build() == 1

    assert not workspace.models_dir.exists()
    assert "cannot read the git commit" in caplog.text


@pytest.mark.unit
def test_build_without_git_is_refused(
    workspace: Workspace, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("PATH", str(workspace.data_dir / "no-programs"))

    assert workspace.build() == 1

    assert not workspace.models_dir.exists()
    assert "git is not installed" in caplog.text


@pytest.mark.unit
def test_dataset_with_an_empty_split_is_refused(
    workspace: Workspace, caplog: pytest.LogCaptureFixture
) -> None:
    (workspace.data_dir / "test" / "images" / "img_b.jpg").unlink()
    (workspace.data_dir / "test" / "test.csv").write_text("image_id\n", encoding="utf-8")

    assert workspace.build() == 1

    assert not workspace.models_dir.exists()
    assert "came out empty" in caplog.text


@pytest.mark.unit
def test_unknown_command_exits_with_usage_error() -> None:
    with pytest.raises(SystemExit) as raised:
        main(["model", "publish"])
    assert raised.value.code == 2
