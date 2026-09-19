"""Builds the model package of an encoder fine-tuned by `finetune --full`."""

import shutil
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Final

import numpy as np
import torch
from numpy.typing import NDArray
from safetensors.torch import load_file

from loadvision.ml.data.facts import photo_facts
from loadvision.ml.data.manifest import Sample
from loadvision.ml.models.backbones import Backbone, load_backbone
from loadvision.ml.packaging.manifest import Manifest, Member, Preprocessing, Runtime
from loadvision.ml.packaging.model_package import (
    PACKAGE_OOF,
    ModelPackage,
    hash_package_files,
    write_package,
)
from loadvision.ml.pipeline.stages.build_package import (
    cv_block,
    write_licenses,
    write_ood,
    write_steps,
)
from loadvision.ml.pipeline.stages.train_folds import Chain, OofRow
from loadvision.ml.training.embedding_cache import read_image
from loadvision.ml.training.finetune import ENCODER_FILE, HEAD_FILE
from loadvision.ml.training.postprocess_fit import fit_chain

_BATCH: Final = 16


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
    ood_percentile: float = 99.0,
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
        ood_percentile: Where the unusual-scene threshold sits among the training scores.

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
    weights = out_dir / "weights" / f"{backbone.spec.timm_name}.finetuned.safetensors"
    weights.parent.mkdir()
    shutil.copyfile(model_dir / ENCODER_FILE, weights)
    head = out_dir / "m0" / HEAD_FILE
    head.parent.mkdir()
    shutil.copyfile(model_dir / HEAD_FILE, head)
    (out_dir / PACKAGE_OOF).parent.mkdir()
    shutil.copyfile(oof_file, out_dir / PACKAGE_OOF)

    raw = np.array([oof[image_id].raw for image_id in ids])
    labels = np.array([oof[image_id].label for image_id in ids])
    steps = write_steps(out_dir, fit_chain(chain, raw, labels, photo_facts(train)))
    ood = write_ood(out_dir, 0, encoder, _embed(backbone, train, device), ood_percentile)
    licenses = write_licenses(out_dir, [backbone], licenses_dir, fine_tuned=True)
    member = Member(
        backbone=encoder,
        runtime=Runtime.TORCH,
        head=str(head.relative_to(out_dir)),
        weights=str(weights.relative_to(out_dir)),
        weight=1.0,
        preprocessing=Preprocessing(name=backbone.preprocessing, params=dict(backbone.params)),
    )
    manifest = Manifest.model_validate(
        {
            "schema_version": 1,
            "version": version,
            "created_at": created_at,
            "git_commit": git_commit,
            "data_hash": data_hash,
            "members": [member.model_dump(mode="json")],
            "postprocess": [step.model_dump(mode="json") for step in steps],
            "licenses": licenses,
            **ood,
            **cv_block(oof_file, folds_file),
            "files": hash_package_files(out_dir),
        }
    )
    return write_package(out_dir, manifest)


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
