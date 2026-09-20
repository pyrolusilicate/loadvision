"""Builds the model package of an encoder fine-tuned by `finetune`, on all photos or by fold."""

import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load_file

from loadvision.imaging.crops import FAR_END, WHOLE
from loadvision.ml.data.facts import photo_facts
from loadvision.ml.data.manifest import Sample
from loadvision.ml.models.backbones import Backbone, load_backbone
from loadvision.ml.models.depth import (
    DEPTH_LICENSE,
    DEPTH_MODEL,
    DEPTH_ORIGIN,
    GRID_COLS,
    GRID_ROWS,
)
from loadvision.ml.packaging.manifest import Manifest, Member, Preprocessing, Runtime
from loadvision.ml.packaging.model_package import (
    PACKAGE_OOF,
    ModelPackage,
    hash_package_files,
    write_package,
)
from loadvision.ml.pipeline.stages.build_package import (
    OOD_PERCENTILE,
    Credit,
    cv_block,
    write_licenses,
    write_ood,
    write_steps,
)
from loadvision.ml.pipeline.stages.train_folds import Chain, OofRow
from loadvision.ml.training.embedding_cache import read_image
from loadvision.ml.training.finetune import ENCODER_FILE, FOLDS_DIR, HEAD_FILE
from loadvision.ml.training.postprocess_fit import fit_chain

_BATCH: Final = 16
_DEPTH_DIR: Final = "depth"
_DEPTH_SOURCE: Final = f"https://huggingface.co/{DEPTH_MODEL}"


def load_finetuned(name: str, encoder: Path, device: str) -> Backbone:
    """Loads a registry encoder with the weights `finetune` saved for it.

    Args:
        name: Registry key the model was fine-tuned from.
        encoder: `encoder.safetensors` of the run.
        device: Torch device.

    Returns:
        The encoder, frozen, with the fine-tuned weights.
    """
    backbone = load_backbone(name, device)
    backbone.module.load_state_dict(load_file(str(encoder), device=device))
    return backbone


def _write_member(
    out_dir: Path,
    index: int,
    encoder: str,
    backbone: Backbone,
    source: Path,
    *,
    weight: float,
    far_view: bool,
) -> Member:
    """Copies one fine-tuned model into the package and describes it for the manifest.

    Args:
        out_dir: Package folder.
        index: Position of the member; its files go into `m<index>`.
        encoder: Registry key of the encoder.
        backbone: That encoder, loaded, for its preprocessing and checkpoint name.
        source: Folder with `encoder.safetensors` and `head.npz` of this model.
        weight: Share of this member in the ensemble average.
        far_view: Whether the model is also shown the enlarged far end of the body.

    Returns:
        The member entry of the manifest.
    """
    weights = out_dir / "weights" / f"{backbone.spec.timm_name}.finetuned.m{index}.safetensors"
    weights.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / ENCODER_FILE, weights)
    head = out_dir / f"m{index}" / HEAD_FILE
    head.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source / HEAD_FILE, head)
    return Member(
        backbone=encoder,
        runtime=Runtime.TORCH,
        head=str(head.relative_to(out_dir)),
        weights=str(weights.relative_to(out_dir)),
        weight=weight,
        preprocessing=Preprocessing(name=backbone.preprocessing, params=dict(backbone.params)),
        inputs=(WHOLE, FAR_END) if far_view else (WHOLE,),
    )


def build_finetuned_package(
    model_dir: Path,
    out_dir: Path,
    *,
    encoder: str,
    train: Sequence[Sample],
    oof: Mapping[str, OofRow],
    oof_file: Path,
    chain: Chain,
    folds_file: Path,
    licenses_dir: Path,
    version: str,
    git_commit: str,
    data_hash: str,
    created_at: datetime,
    device: str,
    ood_percentile: float = OOD_PERCENTILE,
    depth: bool = False,
    far_view: bool = False,
) -> ModelPackage:
    """Writes a one-member package that answers what the fine-tuned model answers.

    The model comes from a run on every photo; its calibration and its metrics come from the
    out-of-fold run of the same settings, because only there every photo was predicted by a
    model that never saw it, the way the test photos are.

    Args:
        model_dir: Folder of the `--full` run with `encoder.safetensors` and `head.npz`.
        out_dir: Package folder; it must not exist yet.
        encoder: Registry key the model was fine-tuned from.
        train: Labelled photos, the ones the model was trained on.
        oof: Rows of the out-of-fold run, by photo id.
        oof_file: `oof.csv` of that run, copied into the package.
        chain: Postprocessing steps to fit on the out-of-fold predictions.
        folds_file: Folds the out-of-fold run was measured on.
        licenses_dir: Folder holding one text per SPDX identifier the registry names.
        version: Package version in the `ModelVersionId` format.
        git_commit: Commit the package is built from, 40 hex characters.
        data_hash: SHA-256 of the dataset manifest.
        created_at: Build time, timezone-aware.
        device: Torch device the encoder runs on while embedding the training photos.
        ood_percentile: Where the unusual-scene threshold sits among the training scores;
            99.9 means one training photo in a thousand would be flagged.
        depth: Whether the model reads a depth grid next to the photo; then the package carries
            the depth checkpoint too.
        far_view: Whether the model is also shown the enlarged far end of the body.

    Returns:
        The written package, loaded back and verified.

    Raises:
        FileExistsError: If `out_dir` already exists.
        FileNotFoundError: If the run lacks a model file or a licence text is missing.
        ValueError: If the out-of-fold run does not cover exactly the training photos.
    """
    ids = [sample.image_id for sample in train]
    if set(oof) != set(ids):
        raise ValueError("the out-of-fold run does not cover exactly the training photos")
    for name in (ENCODER_FILE, HEAD_FILE):
        if not (model_dir / name).is_file():
            raise FileNotFoundError(f"{model_dir / name} is missing; pass a finetune --full run")
    out_dir.mkdir(parents=True)

    backbone = load_finetuned(encoder, model_dir / ENCODER_FILE, device)
    member = _write_member(out_dir, 0, encoder, backbone, model_dir, weight=1.0, far_view=far_view)
    return _finish(
        out_dir,
        [member],
        ood=write_ood(out_dir, 0, encoder, _embed(backbone, train, device), ood_percentile),
        backbone=backbone,
        oof=oof,
        oof_file=oof_file,
        train=train,
        chain=chain,
        folds_file=folds_file,
        licenses_dir=licenses_dir,
        version=version,
        git_commit=git_commit,
        data_hash=data_hash,
        created_at=created_at,
        depth=depth,
    )


def _finish(
    out_dir: Path,
    members: Sequence[Member],
    *,
    ood: Mapping[str, Any],
    backbone: Backbone,
    oof: Mapping[str, OofRow],
    oof_file: Path,
    train: Sequence[Sample],
    chain: Chain,
    folds_file: Path,
    licenses_dir: Path,
    version: str,
    git_commit: str,
    data_hash: str,
    created_at: datetime,
    depth: bool,
) -> ModelPackage:
    """Writes everything a package carries besides its models, then the manifest itself."""
    (out_dir / PACKAGE_OOF).parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(oof_file, out_dir / PACKAGE_OOF)
    ids = [sample.image_id for sample in train]
    raw = np.array([oof[image_id].raw for image_id in ids])
    labels = np.array([oof[image_id].label for image_id in ids])
    steps = write_steps(out_dir, fit_chain(chain, raw, labels, photo_facts(train)))
    extra = [Credit(DEPTH_MODEL, DEPTH_LICENSE, DEPTH_ORIGIN, _DEPTH_SOURCE)] if depth else []
    licenses = write_licenses(out_dir, [backbone], licenses_dir, fine_tuned=True, extra=extra)
    depth_block = {"depth": _place_depth(out_dir)} if depth else {}
    manifest = Manifest.model_validate(
        {
            "schema_version": 1,
            "version": version,
            "created_at": created_at,
            "git_commit": git_commit,
            "data_hash": data_hash,
            "members": [member.model_dump(mode="json") for member in members],
            "postprocess": [step.model_dump(mode="json") for step in steps],
            "licenses": licenses,
            **ood,
            **depth_block,
            **cv_block(oof_file, folds_file),
            "files": hash_package_files(out_dir),
        }
    )
    return write_package(out_dir, manifest)


def build_folds_package(
    run_dir: Path,
    out_dir: Path,
    *,
    encoder: str,
    train: Sequence[Sample],
    oof: Mapping[str, OofRow],
    oof_file: Path,
    chain: Chain,
    folds_file: Path,
    licenses_dir: Path,
    version: str,
    git_commit: str,
    data_hash: str,
    created_at: datetime,
    device: str,
    ood_percentile: float = OOD_PERCENTILE,
    depth: bool = False,
    far_view: bool = False,
) -> ModelPackage:
    """Writes a package of the fold models of one run, averaged in equal shares.

    This is the model a submission of that run's test answers was made with: every fold model
    saw four fifths of the photos, and their average is what answered the test. A package of
    them answers the same way, at the cost of running the encoder once per fold.

    Args:
        run_dir: Folder of the out-of-fold run, with `folds/fold_<n>` and `oof.csv`.
        out_dir: Package folder; it must not exist yet.
        encoder: Registry key the models were fine-tuned from.
        train: Labelled photos the run was measured on.
        oof: Rows of `oof.csv`, by photo id.
        oof_file: That file, copied into the package.
        chain: Postprocessing steps to fit on the out-of-fold predictions.
        folds_file: Folds the run was measured on.
        licenses_dir: Folder holding one text per SPDX identifier the registry names.
        version: Package version in the `ModelVersionId` format.
        git_commit: Commit the package is built from, 40 hex characters.
        data_hash: SHA-256 of the dataset manifest.
        created_at: Build time, timezone-aware.
        device: Torch device the encoders run on while embedding the training photos.
        ood_percentile: Where the unusual-scene threshold sits among the training scores.
        depth: Whether the models read a depth grid next to the photo.
        far_view: Whether they are also shown the enlarged far end of the body.

    Returns:
        The written package, loaded back and verified.

    Raises:
        FileNotFoundError: If the run kept no fold models.
        ValueError: If the run does not cover exactly the training photos.
    """
    ids = [sample.image_id for sample in train]
    if set(oof) != set(ids):
        raise ValueError("the out-of-fold run does not cover exactly the training photos")
    folds = sorted((run_dir / FOLDS_DIR).glob("fold_*"))
    if not folds:
        raise FileNotFoundError(
            f"{run_dir / FOLDS_DIR} has no fold models; the run was made with --no-weights"
        )
    out_dir.mkdir(parents=True)

    members = []
    first: Backbone | None = None
    for index, fold in enumerate(folds):
        backbone = load_finetuned(encoder, fold / ENCODER_FILE, device)
        first = first or backbone
        members.append(
            _write_member(
                out_dir,
                index,
                encoder,
                backbone,
                fold,
                weight=1 / len(folds),
                far_view=far_view,
            )
        )
    assert first is not None  # the folds list is not empty
    return _finish(
        out_dir,
        members,
        # The reference of unusual scenes is the first fold's view of the photos: every fold
        # model sees the same scenes, and one of them is enough to measure the distance.
        ood=write_ood(out_dir, 0, encoder, _embed(first, train, device), ood_percentile),
        backbone=first,
        oof=oof,
        oof_file=oof_file,
        train=train,
        chain=chain,
        folds_file=folds_file,
        licenses_dir=licenses_dir,
        version=version,
        git_commit=git_commit,
        data_hash=data_hash,
        created_at=created_at,
        depth=depth,
    )


def _place_depth(out_dir: Path) -> dict[str, Any]:
    """Copies the depth checkpoint into the package, so the service needs no network.

    Returns:
        The `depth` block of the manifest.
    """
    from huggingface_hub import snapshot_download

    source = Path(snapshot_download(DEPTH_MODEL))
    folder = out_dir / _DEPTH_DIR
    shutil.copytree(source, folder, ignore=shutil.ignore_patterns(".*"))
    return {"model": _DEPTH_DIR, "rows": GRID_ROWS, "cols": GRID_COLS}


def _embed(backbone: Backbone, train: Sequence[Sample], device: str) -> NDArray[np.float64]:
    """Embeds the training photos with the fine-tuned encoder itself.

    The embedding cache is keyed by the registry name, which the fine-tuned encoder shares
    with the pretrained one, so a cached vector would be the pretrained one.
    """
    rows: list[NDArray[np.float64]] = []
    with torch.inference_mode():
        for start in range(0, len(train), _BATCH):
            images = [read_image(sample.path) for sample in train[start : start + _BATCH]]
            rows.append(backbone.embed(images, device).float().cpu().numpy().astype(np.float64))
    return np.concatenate(rows)
