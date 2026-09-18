import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from loadvision.cli.main import main
from loadvision.imaging.prepared_image import MediaType, PreparedImage
from loadvision.ml.inference.predictor import PackagePredictor
from loadvision.ml.packaging.model_package import ModelPackage
from tests.fakes.in_memory_db import InMemoryStore
from tests.memory_database import use_memory_database
from tests.ml.test_dataset_manifest import write_split
from tests.model_packages import build_numpy_package, write_baseline_weights
from tests.settings_files import use_settings

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
    use_settings(
        tmp_path,
        monkeypatch,
        storage={"images_dir": str(tmp_path / "images"), "models_dir": str(tmp_path / "models")},
        model={"version": INITIAL_VERSION},
    )

    baseline_dir = write_baseline_weights(tmp_path / "baseline", load=55.0)

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


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> InMemoryStore:
    return use_memory_database(monkeypatch, "loadvision.cli.commands.model")


def versions(store: InMemoryStore) -> dict[str, str]:
    return {version.value: stored.status.value for version, stored in store.model_versions.items()}


@pytest.mark.unit
def test_registered_package_lands_in_the_models_folder_and_the_database(
    workspace: Workspace, store: InMemoryStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    package = build_numpy_package(tmp_path / "run", load=42.0, version="v1")

    assert main(["model", "register", str(package)]) == 0

    copied = workspace.models_dir / "v1"
    assert (copied / "manifest.json").is_file()
    assert versions(store) == {"v1": "candidate"}
    assert "loadvision model activate v1" in caplog.text


@pytest.mark.unit
def test_package_already_in_the_models_folder_is_registered_where_it_is(
    workspace: Workspace, store: InMemoryStore
) -> None:
    assert workspace.build() == 0
    before = sorted(path.name for path in (workspace.models_dir / INITIAL_VERSION).iterdir())

    assert main(["model", "register", str(workspace.models_dir / INITIAL_VERSION)]) == 0

    assert versions(store) == {INITIAL_VERSION: "candidate"}
    assert (
        sorted(path.name for path in (workspace.models_dir / INITIAL_VERSION).iterdir()) == before
    )


@pytest.mark.unit
def test_another_package_of_the_same_version_is_refused(
    workspace: Workspace, store: InMemoryStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    assert workspace.build() == 0
    other = build_numpy_package(tmp_path / "run", load=42.0, version=INITIAL_VERSION)

    assert main(["model", "register", str(other)]) == 1

    assert "already holds a package" in caplog.text
    assert versions(store) == {}


@pytest.mark.unit
def test_package_the_service_could_not_load_is_not_registered(
    workspace: Workspace, store: InMemoryStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    package = build_numpy_package(tmp_path / "run", load=42.0, version="v1", features=10)

    assert main(["model", "register", str(package)]) == 1

    assert "features" in caplog.text
    assert not (workspace.models_dir / "v1").exists()
    assert versions(store) == {}


@pytest.mark.unit
def test_load_check_can_be_skipped(
    workspace: Workspace, store: InMemoryStore, tmp_path: Path
) -> None:
    package = build_numpy_package(tmp_path / "run", load=42.0, version="v1", features=10)

    assert main(["model", "register", str(package), "--skip-load-check"]) == 0

    assert versions(store) == {"v1": "candidate"}


@pytest.mark.unit
def test_activation_retires_the_previous_version_and_rollback_brings_it_back(
    workspace: Workspace, store: InMemoryStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    for tag in ("v1", "v2"):
        package = build_numpy_package(tmp_path / tag, load=42.0, version=tag)
        assert main(["model", "register", str(package)]) == 0
        assert main(["model", "activate", tag]) == 0

    assert versions(store) == {"v1": "retired", "v2": "active"}

    assert main(["model", "rollback"]) == 0

    assert versions(store) == {"v1": "active", "v2": "retired"}
    assert "rolled back to v1" in caplog.text


@pytest.mark.unit
@pytest.mark.parametrize(
    ("command", "message"),
    [
        (["model", "activate", "v404"], "not registered"),
        (["model", "activate", "../outside"], "invalid version"),
        (["model", "rollback"], "no retired version"),
    ],
    ids=["unknown", "malformed", "nothing-to-roll-back"],
)
def test_activation_problems_are_reported(
    store: InMemoryStore, caplog: pytest.LogCaptureFixture, command: list[str], message: str
) -> None:
    assert main(command) == 1
    assert message in caplog.text


@pytest.mark.unit
def test_list_shows_registered_versions(
    workspace: Workspace, store: InMemoryStore, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)

    assert main(["model", "list"]) == 0
    assert "no versions registered yet" in caplog.text

    package = build_numpy_package(tmp_path / "run", load=42.0, version="v1")
    assert main(["model", "register", str(package)]) == 0
    assert main(["model", "activate", "v1"]) == 0
    caplog.clear()

    assert main(["model", "list"]) == 0

    header, row = (record.getMessage() for record in caplog.records)
    assert header.split() == ["version", "status", "registered", "activated"]
    # Four columns, so no value ran into the next one.
    assert row.split()[:2] == ["v1", "active"]
    assert len(row.split()) == 4
