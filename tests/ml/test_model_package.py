import json
import pickle
from pathlib import Path

import pytest

from loadvision.domain.model_version import ModelVersionId
from loadvision.errors import ErrorCode, LoadvisionError
from loadvision.ml.packaging.manifest import MANIFEST_FILE, Manifest
from loadvision.ml.packaging.model_package import (
    ModelPackage,
    ModelPackageError,
    file_sha256,
    hash_package_files,
    write_package,
)
from tests.model_packages import HEAD_BYTES, build_package, manifest_data, member_data


@pytest.fixture
def root(tmp_path: Path) -> Path:
    return tmp_path / "baseline-v0"


def rewrite_manifest(root: Path, **changes: object) -> None:
    path = root / MANIFEST_FILE
    data = json.loads(path.read_text(encoding="utf-8"))
    data.update(changes)
    path.write_text(json.dumps(data), encoding="utf-8")


def load_error(root: Path) -> ModelPackageError:
    with pytest.raises(ModelPackageError) as caught:
        ModelPackage.load(root)
    return caught.value


@pytest.mark.ml
def test_built_package_loads_with_its_files(root: Path) -> None:
    built = build_package(root)
    loaded = ModelPackage.load(root)
    assert loaded == built
    assert loaded.version == ModelVersionId("baseline-v0")
    assert loaded.path("head.npz").read_bytes() == HEAD_BYTES


@pytest.mark.ml
def test_ignores_files_the_manifest_does_not_list(root: Path) -> None:
    build_package(root)
    (root / "notes.txt").write_text("scratch", encoding="utf-8")
    ModelPackage.load(root)


@pytest.mark.ml
def test_path_rejects_unlisted_name(root: Path) -> None:
    package = build_package(root)
    with pytest.raises(ModelPackageError, match=r"notes\.txt is not listed"):
        package.path("notes.txt")


@pytest.mark.ml
def test_missing_manifest_names_the_file(root: Path) -> None:
    root.mkdir()
    assert MANIFEST_FILE in str(load_error(root).detail)


@pytest.mark.ml
def test_invalid_json_is_a_package_error(root: Path) -> None:
    build_package(root)
    (root / MANIFEST_FILE).write_text("{not json", encoding="utf-8")
    assert "Invalid JSON" in str(load_error(root).detail)


@pytest.mark.ml
def test_schema_error_names_the_field_without_pydantic_links(root: Path) -> None:
    build_package(root)
    rewrite_manifest(root, git_commit="HEAD")
    detail = str(load_error(root).detail)
    assert "git_commit: String should match pattern" in detail
    assert "errors.pydantic.dev" not in detail


@pytest.mark.ml
def test_reports_every_broken_file_together(root: Path) -> None:
    files = {"head.npz": HEAD_BYTES, "m2/head.npz": b"second head"}
    members = [member_data(weight=0.5), member_data(head="m2/head.npz", weight=0.5)]
    build_package(root, files, members=members)
    (root / "head.npz").write_bytes(b"tampered")
    (root / "m2/head.npz").unlink()

    detail = str(load_error(root).detail)

    assert f"head.npz has sha256 {file_sha256(root / 'head.npz')}" in detail
    assert "m2/head.npz is missing" in detail


@pytest.mark.ml
def test_rejects_listed_file_that_resolves_outside_the_package(root: Path, tmp_path: Path) -> None:
    build_package(root)
    outside = tmp_path / "outside.npz"
    outside.write_bytes(HEAD_BYTES)
    (root / "head.npz").unlink()
    (root / "head.npz").symlink_to(outside)
    assert "head.npz points outside the package" in str(load_error(root).detail)


@pytest.mark.ml
def test_listed_directory_is_missing_file(root: Path) -> None:
    build_package(root)
    (root / "head.npz").unlink()
    (root / "head.npz").mkdir()
    assert "head.npz is missing" in str(load_error(root).detail)


@pytest.mark.ml
def test_hash_package_files_lists_nested_files_except_manifest(root: Path) -> None:
    files = {"head.npz": HEAD_BYTES, "m1/backbone.onnx": b"onnx"}
    members = [member_data(runtime="onnx", weights="m1/backbone.onnx")]
    build_package(root, files, members=members)
    assert hash_package_files(root) == {
        "head.npz": file_sha256(root / "head.npz"),
        "m1/backbone.onnx": file_sha256(root / "m1/backbone.onnx"),
    }


@pytest.mark.ml
def test_written_manifest_is_the_parsed_manifest(root: Path) -> None:
    package = build_package(root)
    written = json.loads((root / MANIFEST_FILE).read_text(encoding="utf-8"))
    assert written == json.loads(package.manifest.model_dump_json())
    assert written["created_at"] == manifest_data()["created_at"]


@pytest.mark.ml
def test_write_package_refuses_folder_without_listed_file(root: Path) -> None:
    root.mkdir()
    manifest = Manifest.model_validate(manifest_data())
    with pytest.raises(ModelPackageError, match=r"head\.npz is missing"):
        write_package(root, manifest)


@pytest.mark.ml
def test_package_error_carries_catalog_code_and_survives_pickling() -> None:
    error = ModelPackageError("head.npz is missing")
    restored = pickle.loads(pickle.dumps(error))
    assert isinstance(restored, LoadvisionError)
    assert type(restored) is ModelPackageError
    assert (restored.code, restored.detail) == (ErrorCode.MODEL_PACKAGE_INVALID, error.detail)
