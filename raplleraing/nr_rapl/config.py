"""Configuration objects for the optional NR_RAPL path.

RAPL uses argparse rather than a YAML/YACS configuration system.  These
dataclasses provide the requested nested ``METHOD.NAME`` and ``NR_RAPL``
interface while ``build_method_config`` adapts existing command-line args.
"""

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Dict, List, Optional


@dataclass
class MethodNameConfig:
    NAME: str = "RAPL"


@dataclass
class CropConfig:
    ENABLED: bool = False
    NUM_CANDIDATES: int = 8
    NUM_POSITIVE: int = 1
    NUM_NEGATIVE: int = 1
    MIN_SCALE: float = 0.08
    MAX_SCALE: float = 1.0
    SELECTION: str = "margin"
    ENCODE_BATCH_SIZE: int = 16
    NEGATIVE_MARGIN_THRESHOLD: Optional[float] = None
    MIN_MARGIN_SPREAD: float = 1e-6


@dataclass
class NegativePromptConfig:
    ENABLED: bool = False
    NUM: int = 100
    N_CTX: Optional[int] = None
    INIT: str = "random"
    CLASS_AGNOSTIC: bool = True
    TEMPLATE: str = "object"
    INIT_NOISE_STD: float = 0.01
    DIVERSITY_CHUNK_SIZE: int = 0


@dataclass
class LossConfig:
    LAMBDA_POS: float = 1.0
    LAMBDA_NEG: float = 1.0
    LAMBDA_DIV: float = 0.05
    NEG_WARMUP_EPOCHS: int = 5
    KEEP_ORIGINAL_LOCAL_LOSS: bool = True
    LAMBDA_LOCAL_ORIGINAL: float = 1.0


@dataclass
class OODScoreConfig:
    TYPE: str = "NR_GLMCM"
    NEG_DENOM_WEIGHT: float = 1.0
    LOCAL_WEIGHT: float = 1.0
    TEST_TOPK: Optional[List[int]] = None


@dataclass
class ClassificationConfig:
    USE_NEGATIVE_CALIBRATION: bool = False


@dataclass
class OptimizerConfig:
    GLOBAL_LR_MULTIPLIER: float = 1.0
    LOCAL_POSITIVE_LR_MULTIPLIER: float = 1.0
    LOCAL_NEGATIVE_LR_MULTIPLIER: float = 1.0
    LOCAL_PROJECTION_LR_MULTIPLIER: float = 1.0


@dataclass
class AblationConfig:
    USE_GUIDED_CROP: bool = False
    USE_MARGIN_SELECTION: bool = True
    USE_NEG_PROMPT: bool = False
    USE_NEG_LOSS: bool = False
    USE_DIV_LOSS: bool = False
    USE_NR_GLMCM: bool = False


@dataclass
class NRRAPLConfig:
    ENABLED: bool = False
    TRAIN_STAGE: str = "bootstrap"
    CROP: CropConfig = field(default_factory=CropConfig)
    NEG_PROMPT: NegativePromptConfig = field(default_factory=NegativePromptConfig)
    LOSS: LossConfig = field(default_factory=LossConfig)
    OOD_SCORE: OODScoreConfig = field(default_factory=OODScoreConfig)
    CLASSIFICATION: ClassificationConfig = field(default_factory=ClassificationConfig)
    OPTIM: OptimizerConfig = field(default_factory=OptimizerConfig)
    RESUME_OPTIMIZER: bool = False
    ABLATION: AblationConfig = field(default_factory=AblationConfig)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class MethodConfig:
    METHOD: MethodNameConfig = field(default_factory=MethodNameConfig)
    NR_RAPL: NRRAPLConfig = field(default_factory=NRRAPLConfig)


def add_NR_RAPL_arguments(parser: Any, boolean_type: Any) -> None:
    """Register optional NR_RAPL flags without changing legacy defaults."""
    parser.add_argument("--method_name", default="RAPL", choices=["RAPL", "NR_RAPL"], type=str)
    parser.add_argument("--nr_enabled", default=False, type=boolean_type)
    parser.add_argument("--nr_train_stage", default="bootstrap", type=str)

    parser.add_argument("--nr_crop_enabled", default=False, type=boolean_type)
    parser.add_argument("--nr_crop_num_candidates", default=8, type=int)
    parser.add_argument("--nr_crop_num_positive", default=1, type=int)
    parser.add_argument("--nr_crop_num_negative", default=1, type=int)
    parser.add_argument("--nr_crop_min_scale", default=0.08, type=float)
    parser.add_argument("--nr_crop_max_scale", default=1.0, type=float)
    parser.add_argument("--nr_crop_selection", default="margin", choices=["margin", "similarity"], type=str)
    parser.add_argument("--nr_crop_encode_batch_size", default=16, type=int)
    parser.add_argument("--nr_crop_negative_margin_threshold", default=None, type=float)
    parser.add_argument("--nr_crop_min_margin_spread", default=1e-6, type=float)

    parser.add_argument("--nr_neg_prompt_enabled", default=False, type=boolean_type)
    parser.add_argument("--nr_neg_prompt_num", default=100, type=int)
    parser.add_argument("--nr_neg_prompt_n_ctx", default=None, type=int)
    parser.add_argument("--nr_neg_prompt_init", default="random", choices=["random", "local_mean", "text"], type=str)
    parser.add_argument("--nr_neg_prompt_class_agnostic", default=True, type=boolean_type)
    parser.add_argument("--nr_neg_prompt_template", default="object", type=str)
    parser.add_argument("--nr_neg_prompt_init_noise_std", default=0.01, type=float)
    parser.add_argument("--nr_neg_prompt_diversity_chunk_size", default=0, type=int)

    parser.add_argument("--nr_lambda_pos", default=1.0, type=float)
    parser.add_argument("--nr_lambda_neg", default=1.0, type=float)
    parser.add_argument("--nr_lambda_div", default=0.05, type=float)
    parser.add_argument("--nr_neg_warmup_epochs", default=5, type=int)
    parser.add_argument("--nr_keep_original_local_loss", default=True, type=boolean_type)
    parser.add_argument("--nr_lambda_local_original", default=1.0, type=float)

    parser.add_argument("--nr_ood_score_type", default="NR_GLMCM", type=str)
    parser.add_argument("--nr_neg_denom_weight", default=1.0, type=float)
    parser.add_argument("--nr_local_weight", default=1.0, type=float)
    parser.add_argument("--nr_ood_test_topk", default=None, type=int, nargs="+")
    parser.add_argument("--nr_use_negative_calibration", default=False, type=boolean_type)
    parser.add_argument("--nr_global_lr_multiplier", default=1.0, type=float)
    parser.add_argument("--nr_local_positive_lr_multiplier", default=1.0, type=float)
    parser.add_argument("--nr_local_negative_lr_multiplier", default=1.0, type=float)
    parser.add_argument("--nr_local_projection_lr_multiplier", default=1.0, type=float)
    parser.add_argument("--nr_resume_optimizer", default=False, type=boolean_type)

    parser.add_argument("--nr_use_guided_crop", default=False, type=boolean_type)
    # None preserves the existing --nr_crop_selection option for callers that
    # predate the explicit ablation switch.
    parser.add_argument("--nr_use_margin_selection", default=None, type=boolean_type)
    parser.add_argument("--nr_use_neg_prompt", default=False, type=boolean_type)
    parser.add_argument("--nr_use_neg_loss", default=False, type=boolean_type)
    parser.add_argument("--nr_use_div_loss", default=False, type=boolean_type)
    parser.add_argument("--nr_use_nr_glmcm", default=False, type=boolean_type)


def build_method_config(args: Any) -> MethodConfig:
    """Adapt parser arguments to the nested configuration used by NR_RAPL."""
    config = MethodConfig()
    config.METHOD.NAME = args.method_name
    nr = config.NR_RAPL
    nr.ENABLED = bool(args.nr_enabled or args.method_name == "NR_RAPL")
    nr.TRAIN_STAGE = args.nr_train_stage
    nr.CROP = CropConfig(
        ENABLED=args.nr_crop_enabled,
        NUM_CANDIDATES=args.nr_crop_num_candidates,
        NUM_POSITIVE=args.nr_crop_num_positive,
        NUM_NEGATIVE=args.nr_crop_num_negative,
        MIN_SCALE=args.nr_crop_min_scale,
        MAX_SCALE=args.nr_crop_max_scale,
        SELECTION=args.nr_crop_selection,
        ENCODE_BATCH_SIZE=args.nr_crop_encode_batch_size,
        NEGATIVE_MARGIN_THRESHOLD=args.nr_crop_negative_margin_threshold,
        MIN_MARGIN_SPREAD=args.nr_crop_min_margin_spread,
    )
    nr.NEG_PROMPT = NegativePromptConfig(
        ENABLED=args.nr_neg_prompt_enabled,
        NUM=args.nr_neg_prompt_num,
        N_CTX=args.nr_neg_prompt_n_ctx,
        INIT=args.nr_neg_prompt_init,
        CLASS_AGNOSTIC=args.nr_neg_prompt_class_agnostic,
        TEMPLATE=args.nr_neg_prompt_template,
        INIT_NOISE_STD=args.nr_neg_prompt_init_noise_std,
        DIVERSITY_CHUNK_SIZE=args.nr_neg_prompt_diversity_chunk_size,
    )
    nr.LOSS = LossConfig(
        LAMBDA_POS=args.nr_lambda_pos,
        LAMBDA_NEG=args.nr_lambda_neg,
        LAMBDA_DIV=args.nr_lambda_div,
        NEG_WARMUP_EPOCHS=args.nr_neg_warmup_epochs,
        KEEP_ORIGINAL_LOCAL_LOSS=args.nr_keep_original_local_loss,
        LAMBDA_LOCAL_ORIGINAL=args.nr_lambda_local_original,
    )
    nr.OOD_SCORE = OODScoreConfig(
        TYPE=args.nr_ood_score_type,
        NEG_DENOM_WEIGHT=args.nr_neg_denom_weight,
        LOCAL_WEIGHT=args.nr_local_weight,
        TEST_TOPK=args.nr_ood_test_topk,
    )
    nr.CLASSIFICATION = ClassificationConfig(USE_NEGATIVE_CALIBRATION=args.nr_use_negative_calibration)
    nr.OPTIM = OptimizerConfig(
        GLOBAL_LR_MULTIPLIER=args.nr_global_lr_multiplier,
        LOCAL_POSITIVE_LR_MULTIPLIER=args.nr_local_positive_lr_multiplier,
        LOCAL_NEGATIVE_LR_MULTIPLIER=args.nr_local_negative_lr_multiplier,
        LOCAL_PROJECTION_LR_MULTIPLIER=args.nr_local_projection_lr_multiplier,
    )
    nr.RESUME_OPTIMIZER = args.nr_resume_optimizer
    use_margin_selection = (
        args.nr_crop_selection == "margin"
        if args.nr_use_margin_selection is None else args.nr_use_margin_selection
    )
    nr.ABLATION = AblationConfig(
        USE_GUIDED_CROP=args.nr_use_guided_crop,
        USE_MARGIN_SELECTION=use_margin_selection,
        USE_NEG_PROMPT=args.nr_use_neg_prompt,
        USE_NEG_LOSS=args.nr_use_neg_loss,
        USE_DIV_LOSS=args.nr_use_div_loss,
        USE_NR_GLMCM=args.nr_use_nr_glmcm,
    )
    # The explicit flag is authoritative while preserving the original string
    # configuration in every saved experiment snapshot.
    nr.CROP.SELECTION = "margin" if nr.ABLATION.USE_MARGIN_SELECTION else "similarity"
    if config.METHOD.NAME == "NR_RAPL":
        needs_negative_prompts = nr.NEG_PROMPT.ENABLED and nr.ABLATION.USE_NEG_PROMPT
        if nr.CROP.ENABLED and nr.ABLATION.USE_GUIDED_CROP:
            if nr.CROP.NUM_CANDIDATES < 2:
                raise ValueError("Guided crop selection requires CROP.NUM_CANDIDATES >= 2")
            if nr.CROP.NUM_POSITIVE != 1:
                raise ValueError("Only one positive crop is supported; set CROP.NUM_POSITIVE=1")
            if nr.CROP.NUM_NEGATIVE < 1:
                raise ValueError("CROP.NUM_NEGATIVE must be >= 1")
            if not (0.0 < nr.CROP.MIN_SCALE <= nr.CROP.MAX_SCALE <= 1.0):
                raise ValueError("CROP.MIN_SCALE and CROP.MAX_SCALE must satisfy 0 < min <= max <= 1")
        if needs_negative_prompts and nr.NEG_PROMPT.NUM < 1:
            raise ValueError("Enabled negative prompts require NEG_PROMPT.NUM >= 1")
        if nr.ABLATION.USE_NEG_LOSS and not needs_negative_prompts:
            raise ValueError("USE_NEG_LOSS requires enabled USE_NEG_PROMPT and NEG_PROMPT.ENABLED")
        if nr.ABLATION.USE_DIV_LOSS and not needs_negative_prompts:
            raise ValueError("USE_DIV_LOSS requires enabled USE_NEG_PROMPT and NEG_PROMPT.ENABLED")
        for name, value in {
            "LAMBDA_POS": nr.LOSS.LAMBDA_POS,
            "LAMBDA_NEG": nr.LOSS.LAMBDA_NEG,
            "LAMBDA_DIV": nr.LOSS.LAMBDA_DIV,
            "NEG_DENOM_WEIGHT": nr.OOD_SCORE.NEG_DENOM_WEIGHT,
            "LOCAL_WEIGHT": nr.OOD_SCORE.LOCAL_WEIGHT,
        }.items():
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be a finite non-negative value, got {value}")
    return config
