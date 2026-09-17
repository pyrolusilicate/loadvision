"""Where commands write: run folders, the cache between runs and research records."""

from pathlib import Path
from typing import Final

from loadvision.ml.pipeline.stages.extract_embeddings import EMBEDDINGS_DIR
from loadvision.settings import load_settings

CACHE_SUBDIR: Final = "cache"
PROBES_FILE: Final = Path("probes/encoders.jsonl")
OOF_HEADS_FILE: Final = Path("oof/history.jsonl")
OOF_POSTPROCESS_FILE: Final = Path("oof/postprocess.jsonl")
OOF_PREDICTIONS_FILE: Final = Path("oof/predictions.csv")
ERROR_REPORT_FILE: Final = Path("eda/errors.html")


def runs_dir(override: Path | None = None) -> Path:
    """Returns the folder that holds run folders and research records.

    Args:
        override: Folder given on the command line; the settings decide when it is None.
    """
    return load_settings().runs.dir if override is None else override


def cache_dir(runs: Path) -> Path:
    """Returns the cache kept between runs, outside every run folder."""
    return runs / CACHE_SUBDIR


def embeddings_cache(runs: Path) -> Path:
    """Returns the folder with cached embeddings, the one the pipeline stages use."""
    return cache_dir(runs) / EMBEDDINGS_DIR
