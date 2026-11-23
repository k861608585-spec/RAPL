"""Reproducible NR_RAPL ablation metadata and fixed-schema CSV reporting."""

from __future__ import annotations

import csv
import json
import subprocess
import warnings
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


# Stable columns make rows directly consumable by table/plot scripts. The JSON
# column also preserves results from custom OOD loaders without schema drift.
RESULT_COLUMNS = [
    "experiment_name", "ablation_name", "method_name", "seed", "dataset_name", "num_shots", "backbone",
    "global_prompts", "local_prompts", "local_scales", "use_guided_crop", "use_margin_selection",
    "use_neg_prompt", "use_neg_loss", "use_div_loss", "keep_original_local_loss", "use_nr_glmcm",
    "ood_score_type", "neg_prompt_num", "crop_num_candidates", "crop_num_negative", "lambda_neg",
    "lambda_div", "id_accuracy", "ood_inaturalist_fpr95", "ood_inaturalist_auroc", "ood_sun_fpr95",
    "ood_sun_auroc", "ood_places_fpr95", "ood_places_auroc", "ood_textures_fpr95", "ood_textures_auroc",
    "ood_metrics_json", "mean_fpr95", "mean_auroc", "best_epoch", "train_seconds", "test_seconds",
    "max_memory_gb", "git_commit", "checkpoint_path", "config_path",
]


def _git_commit(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError) as error:
        warnings.warn(f"Unable to resolve Git commit hash: {error}", RuntimeWarning, stacklevel=2)
        return "unknown"


def _json_default(value: Any) -> Any:
    return asdict(value) if is_dataclass(value) else str(value)


class ExperimentReporter:
    """Persist one full config and append one result row per invocation."""

    def __init__(self, args: Any, method_config: Any) -> None:
        self.args = args
        self.method_config = method_config
        self.output_dir = Path(args.save_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.output_dir / "experiment_config.json"
        self.csv_path = Path(args.results_csv) if args.results_csv else self.output_dir / "ablation_results.csv"
        self.repo_root = Path(__file__).resolve().parents[2]
        self.git_commit = _git_commit(self.repo_root)

    def save_configuration(self) -> None:
        payload = {
            "args": vars(self.args).copy(),
            "method": self.method_config.METHOD.NAME,
            "method_config": (
                self.method_config.NR_RAPL.to_dict()
                if self.method_config.METHOD.NAME == "NR_RAPL" else {"ENABLED": False}
            ),
            "git_commit": self.git_commit,
        }
        with self.config_path.open("w", encoding="utf-8") as file:
            json.dump(payload, file, indent=2, sort_keys=True, default=_json_default)

    def log_modules(self, logger: Any) -> None:
        if self.method_config.METHOD.NAME == "RAPL":
            logger.info("Ablation modules: METHOD=RAPL | guided_crop=False | neg_prompt=False | neg_loss=False | div_loss=False | NR_GLMCM=False")
            return
        nr = self.method_config.NR_RAPL
        ablation = nr.ABLATION
        logger.info(
            "Ablation modules: METHOD=NR_RAPL | guided_crop=%s | margin_selection=%s | neg_prompt=%s | "
            "neg_loss=%s | div_loss=%s | keep_original_local_loss=%s | NR_GLMCM=%s | score=%s",
            ablation.USE_GUIDED_CROP, ablation.USE_MARGIN_SELECTION,
            bool(nr.NEG_PROMPT.ENABLED and ablation.USE_NEG_PROMPT),
            ablation.USE_NEG_LOSS, ablation.USE_DIV_LOSS, nr.LOSS.KEEP_ORIGINAL_LOCAL_LOSS,
            ablation.USE_NR_GLMCM, nr.OOD_SCORE.TYPE,
        )

    @staticmethod
    def _ood_value(metrics: Mapping[str, Mapping[str, float]], dataset: str, metric: str) -> Optional[float]:
        for name, values in metrics.items():
            if name.lower() == dataset.lower():
                return values.get(metric)
        return None

    def append_result(
        self,
        id_accuracy: Optional[float],
        ood_metrics: Optional[Mapping[str, Mapping[str, float]]],
        best_epoch: Optional[int],
        train_seconds: float,
        test_seconds: float,
        max_memory_gb: float,
        checkpoint_path: Optional[str],
    ) -> None:
        metrics = ood_metrics or {}
        fprs = [float(value["fpr95"]) for value in metrics.values() if "fpr95" in value]
        aurocs = [float(value["auroc"]) for value in metrics.values() if "auroc" in value]
        is_nr = self.method_config.METHOD.NAME == "NR_RAPL"
        nr = self.method_config.NR_RAPL
        ablation = nr.ABLATION
        row: Dict[str, Any] = {
            "experiment_name": self.args.exp_name or self.args.ablation_name or "unnamed",
            "ablation_name": self.args.ablation_name or "none",
            "method_name": self.method_config.METHOD.NAME,
            "seed": self.args.seed, "dataset_name": self.args.dataset_name, "num_shots": self.args.num_shots,
            "backbone": self.args.clip_name, "global_prompts": self.args.n_global_prompts,
            "local_prompts": self.args.n_local_prompts, "local_scales": json.dumps(self.args.topk),
            "use_guided_crop": bool(is_nr and ablation.USE_GUIDED_CROP),
            "use_margin_selection": bool(is_nr and ablation.USE_MARGIN_SELECTION),
            "use_neg_prompt": bool(is_nr and nr.NEG_PROMPT.ENABLED and ablation.USE_NEG_PROMPT),
            "use_neg_loss": bool(is_nr and ablation.USE_NEG_LOSS),
            "use_div_loss": bool(is_nr and ablation.USE_DIV_LOSS),
            "keep_original_local_loss": bool(is_nr and nr.LOSS.KEEP_ORIGINAL_LOCAL_LOSS),
            "use_nr_glmcm": bool(is_nr and ablation.USE_NR_GLMCM),
            "ood_score_type": nr.OOD_SCORE.TYPE if is_nr else self.args.ood_method,
            "neg_prompt_num": nr.NEG_PROMPT.NUM if is_nr else 0,
            "crop_num_candidates": nr.CROP.NUM_CANDIDATES if is_nr else 0,
            "crop_num_negative": nr.CROP.NUM_NEGATIVE if is_nr else 0,
            "lambda_neg": nr.LOSS.LAMBDA_NEG if is_nr else 0.0,
            "lambda_div": nr.LOSS.LAMBDA_DIV if is_nr else 0.0,
            "id_accuracy": id_accuracy,
            "ood_inaturalist_fpr95": self._ood_value(metrics, "iNaturalist", "fpr95"),
            "ood_inaturalist_auroc": self._ood_value(metrics, "iNaturalist", "auroc"),
            "ood_sun_fpr95": self._ood_value(metrics, "SUN", "fpr95"),
            "ood_sun_auroc": self._ood_value(metrics, "SUN", "auroc"),
            "ood_places_fpr95": self._ood_value(metrics, "Places", "fpr95"),
            "ood_places_auroc": self._ood_value(metrics, "Places", "auroc"),
            "ood_textures_fpr95": self._ood_value(metrics, "Textures", "fpr95"),
            "ood_textures_auroc": self._ood_value(metrics, "Textures", "auroc"),
            "ood_metrics_json": json.dumps(metrics, sort_keys=True),
            "mean_fpr95": sum(fprs) / len(fprs) if fprs else None,
            "mean_auroc": sum(aurocs) / len(aurocs) if aurocs else None,
            "best_epoch": best_epoch, "train_seconds": train_seconds, "test_seconds": test_seconds,
            "max_memory_gb": max_memory_gb, "git_commit": self.git_commit,
            "checkpoint_path": checkpoint_path or "", "config_path": str(self.config_path),
        }
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists() or self.csv_path.stat().st_size == 0
        with self.csv_path.open("a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=RESULT_COLUMNS, extrasaction="ignore")
            if write_header:
                writer.writeheader()
            writer.writerow(row)
