import json
import re
from typing import Any

import pytest
from pydantic import ValidationError

from loadvision.domain.model_version import ModelVersionId
from loadvision.ml.packaging.manifest import Manifest, Runtime
from tests.builders import T0
from tests.model_packages import manifest_data, member_data

SHA_A = "a" * 64


def parse(data: dict[str, Any]) -> Manifest:
    # The loader reads JSON text, so tests go through the same JSON parsing.
    return Manifest.model_validate_json(json.dumps(data))


def assert_rejected(data: dict[str, Any], match: str) -> None:
    with pytest.raises(ValidationError, match=match):
        parse(data)


@pytest.mark.ml
def test_valid_manifest_is_parsed() -> None:
    manifest = parse(manifest_data())
    assert manifest.model_version == ModelVersionId("baseline-v0")
    assert manifest.created_at == T0
    assert manifest.members[0].runtime is Runtime.NUMPY
    assert (manifest.cv, manifest.ood, manifest.folds, manifest.licenses) == (None, None, None, ())
    assert (manifest.postprocess.round_to, manifest.postprocess.snap_100) == (None, None)


@pytest.mark.ml
def test_written_manifest_parses_back_with_utc_time() -> None:
    manifest = parse(manifest_data(created_at="2026-09-16T15:00:00+03:00"))
    text = manifest.model_dump_json()
    assert json.loads(text)["created_at"] == "2026-09-16T12:00:00Z"
    assert Manifest.model_validate_json(text) == manifest


@pytest.mark.ml
def test_full_manifest_is_parsed() -> None:
    files = {"m1/backbone.onnx": SHA_A, "m1/head.npz": SHA_A, "m2/backbone.onnx": SHA_A}
    files |= {"m2/head.npz": SHA_A, "ood/train.npy": SHA_A, "licenses/DINOv3.md": SHA_A}
    data = manifest_data(
        members=[
            member_data(
                backbone="dinov3_vitb16",
                runtime="onnx",
                weights="m1/backbone.onnx",
                head="m1/head.npz",
                weight=0.6,
                preprocessing={"name": "resize_pad_3x4", "params": {"height": 512, "width": 384}},
            ),
            member_data(
                backbone="siglip2_b16",
                runtime="torch",
                weights="m2/backbone.onnx",
                head="m2/head.npz",
                weight=0.4,
            ),
        ],
        cv={"mae": 7.9, "within_10": 81.5, "mae_by_range": {"90-99": 9.1, "100": 3.2}},
        ood={"k": 10, "threshold": 0.35, "reference": "ood/train.npy"},
        postprocess={"round_to": 5, "snap_100": 97.5},
        folds={"file": "configs/splits/folds.json", "sha256": SHA_A},
        licenses=["licenses/DINOv3.md"],
        files=files,
    )
    manifest = parse(data)
    assert manifest.referenced_files() == [
        "m1/backbone.onnx",
        "m1/head.npz",
        "m2/backbone.onnx",
        "m2/head.npz",
        "ood/train.npy",
        "licenses/DINOv3.md",
    ]
    assert manifest.members[0].preprocessing.params == {"height": 512, "width": 384}


@pytest.mark.ml
@pytest.mark.parametrize("schema_version", [None, 0, 2, "1"])
def test_rejects_other_schema_versions(schema_version: object) -> None:
    data = manifest_data(schema_version=schema_version)
    if schema_version is None:
        del data["schema_version"]
    assert_rejected(data, "schema_version")


@pytest.mark.ml
@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("version", ".hidden"),
        ("version", "v1/../../etc"),
        ("created_at", "2026-09-16T12:00:00"),
        ("git_commit", "0123456"),
        ("git_commit", "0123456789ABCDEF0123456789ABCDEF01234567"),
        ("data_hash", "sha256:" + SHA_A),
        ("members", []),
    ],
)
def test_rejects_malformed_top_level_field(field: str, value: object) -> None:
    assert_rejected(manifest_data(**{field: value}), field)


@pytest.mark.ml
@pytest.mark.parametrize(
    "extra",
    [{"clip": True}, {"postprocess": {"round_to": 5, "clip": True}}],
)
def test_rejects_unknown_fields(extra: dict[str, Any]) -> None:
    assert_rejected(manifest_data(**extra), "Extra inputs are not permitted")


@pytest.mark.ml
@pytest.mark.parametrize("weights", [[0.5, 0.4], [0.7, 0.4], [0.6, 0.400002]])
def test_rejects_member_weights_not_summing_to_one(weights: list[float]) -> None:
    members = [member_data(weight=weight) for weight in weights]
    assert_rejected(manifest_data(members=members), "sum to 1")


@pytest.mark.ml
def test_accepts_weight_sum_within_rounding_tolerance() -> None:
    members = [member_data(weight=0.6), member_data(weight=0.4000004)]
    parse(manifest_data(members=members))


@pytest.mark.ml
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"weight": -0.1}, "weight"),
        ({"weight": 1.5}, "weight"),
        ({"runtime": "tensorrt"}, "runtime"),
        ({"runtime": "onnx"}, "needs a weights file"),
        ({"backbone": "DINOv3"}, "backbone"),
        ({"preprocessing": {"name": "resize", "params": {"pad": "constant"}}}, "params"),
        ({"preprocessing": {"name": "resize", "params": {"crop": True}}}, "params"),
        ({"preprocessing": {"name": "Resize 3x4"}}, "name"),
    ],
)
def test_rejects_malformed_member(overrides: dict[str, Any], match: str) -> None:
    assert_rejected(manifest_data(members=[member_data(**overrides)]), match)


@pytest.mark.ml
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_rejects_non_finite_numbers(value: float) -> None:
    # params have no bounds, so only the finiteness rule can reject the value.
    preprocessing = {"name": "resize", "params": {"scale": value}}
    text = json.dumps(manifest_data(members=[member_data(preprocessing=preprocessing)]))
    with pytest.raises(ValidationError, match="params"):
        Manifest.model_validate_json(text)


@pytest.mark.ml
@pytest.mark.parametrize(
    "path",
    [
        "",
        "/etc/passwd",
        "../head.npz",
        "m1/../../head.npz",
        "./head.npz",
        "m1/",
        "m1//head.npz",
        ".hidden",
        "m1/.hidden",
        "m1\\head.npz",
        "m1/head.npz\n",
        "manifest.json",
    ],
)
def test_rejects_package_path_leaving_the_package(path: str) -> None:
    data = manifest_data(members=[member_data(head=path)], files={path: SHA_A})
    with pytest.raises(ValidationError):
        parse(data)


@pytest.mark.ml
@pytest.mark.parametrize(
    ("overrides", "missing"),
    [
        ({"members": [member_data(head="m1/head.npz")]}, "m1/head.npz"),
        (
            {"members": [member_data(runtime="onnx", weights="m1/backbone.onnx")]},
            "m1/backbone.onnx",
        ),
        ({"ood": {"k": 10, "threshold": 0.3, "reference": "ood/train.npy"}}, "ood/train.npy"),
        ({"licenses": ["licenses/DINOv3.md"]}, "licenses/DINOv3.md"),
    ],
)
def test_rejects_referenced_file_missing_from_files(
    overrides: dict[str, Any], missing: str
) -> None:
    assert_rejected(manifest_data(**overrides), re.escape(f"files not listed in files: {missing}"))


@pytest.mark.ml
@pytest.mark.parametrize(
    "overrides",
    [
        {"cv": {"mae": 7.9, "within_10": 81.5}},
        {"folds": {"file": "configs/splits/folds.json", "sha256": SHA_A}},
    ],
)
def test_requires_cv_and_folds_together(overrides: dict[str, Any]) -> None:
    assert_rejected(manifest_data(**overrides), "cv and folds must be set together")


@pytest.mark.ml
@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"cv": {"mae": -1, "within_10": 80}}, "mae"),
        ({"cv": {"mae": 7.9, "within_10": 815}}, "within_10"),
        ({"cv": {"mae": 7.9, "within_10": 80, "mae_by_range": {"100": -3}}}, "mae_by_range"),
        ({"folds": {"file": "../folds.json", "sha256": SHA_A}}, "folds.file"),
        ({"folds": {"file": "configs/splits/folds.json", "sha256": "abc"}}, "folds.sha256"),
        ({"ood": {"k": 0, "threshold": 0.3, "reference": "head.npz"}}, "ood.k"),
        ({"postprocess": {"round_to": 0}}, "round_to"),
        ({"postprocess": {"snap_100": 101}}, "snap_100"),
    ],
)
def test_rejects_malformed_sections(overrides: dict[str, Any], match: str) -> None:
    data = manifest_data(
        cv={"mae": 7.9, "within_10": 80}, folds={"file": "f.json", "sha256": SHA_A}
    )
    data.update(overrides)
    assert_rejected(data, match)


@pytest.mark.ml
def test_json_schema_can_be_generated_for_documentation() -> None:
    schema = Manifest.model_json_schema()
    assert set(schema["required"]) == {
        "schema_version",
        "version",
        "created_at",
        "git_commit",
        "data_hash",
        "members",
        "files",
    }
