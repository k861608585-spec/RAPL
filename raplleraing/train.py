from typing import Type, Dict, Tuple, Optional
from collections import defaultdict
import os
import sys
import math
import argparse
import time

# Permit both supported module execution (``python -m RAPL.train``) and the
# legacy direct-file form (``python RAPL/train.py``) from any working dir.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import torch
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import _LRScheduler, CosineAnnealingLR
from clip.clip import _transform
from timm.utils import accuracy

import RAPL.lib as lib
import RAPL.vlprompt.tools as vlp_tools
import RAPL.datasets.tools as dts_tools
from RAPL.datasets.tools.guided_crops import (
    GuidedCropDataset,
    make_guided_crop_candidates,
    maybe_wrap_guided_crop_dataset,
)
from RAPL.datasets import return_train_val_datasets, return_ood_loaders, return_domains_loaders
from RAPL.vlprompt import RAPL, NRRAPL
from RAPL.vlprompt.tools import GlobalLocalLoss
from RAPL.nr_rapl.config import add_NR_RAPL_arguments, build_method_config
from RAPL.nr_rapl.experiment import ExperimentReporter
from RAPL.nr_rapl.optimizers import get_NR_RAPL_optimizer

NoneType = Type[None]


def _create_grad_scaler(enabled: bool) -> GradScaler:
    """Create an AMP scaler on both recent and legacy PyTorch releases."""
    # PyTorch 2.6 deprecates torch.cuda.amp.GradScaler in favour of this API.
    # The fallback preserves compatibility with the project's PyTorch 1.x env.
    torch_amp = getattr(torch, "amp", None)
    if torch_amp is not None and hasattr(torch_amp, "GradScaler"):
        try:
            return torch_amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            pass
    return GradScaler(enabled=enabled)


def train_one_epoch(
    model: RAPL,
    train_loader: DataLoader,
    loss_fn: GlobalLocalLoss,
    optimizer: Optimizer,
    lr_scheduler: _LRScheduler,
    epoch: int,
    fp16_scaler: GradScaler,
    args: argparse.Namespace,
) -> lib.DictAverage:
    meter = lib.DictAverage()
    progress = lib.ProgressMeter(len(train_loader), meter, prefix=f"Epoch: [{epoch}]")

    class_names = train_loader.dataset.all_names

    if not args.learn_global_prompt and not args.learn_local_prompts:
        with torch.no_grad(), autocast(enabled=args.use_fp16):
            text_features, local_text_features = model.encode_text(class_names)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            local_text_features /= local_text_features.norm(dim=-1, keepdim=True)
    else:
        text_features = local_text_features = None

    model.train()
    optimizer.zero_grad()
    track_loader = lib.track(train_loader, f"Epoch {epoch} / {args.max_epoch}")
    for i, batch in enumerate(track_loader):
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)
        guided_selection = None
        # ``guided_crops`` is intentionally absent from the original RAPL
        # dataset pipeline.  Keep that pipeline byte-for-byte optional: the
        # selector is reachable only for an explicitly enabled NR_RAPL run.
        use_guided_crop = (
            args.method_name == "NR_RAPL"
            and isinstance(model, NRRAPL)
            and args.nr_crop_enabled
            and args.nr_use_guided_crop
            and model.uses_guided_crops
        )
        if use_guided_crop:
            if "guided_crops" not in batch:
                # Compatibility fallback for custom/stale dataset wrappers:
                # ImageNet-style datasets retain their unmodified paths in the
                # batch, so candidates can still be built on CPU with exactly
                # the same crop/normalisation implementation.
                paths = batch.get("path")
                if not isinstance(paths, (list, tuple)) or len(paths) != images.shape[0]:
                    raise RuntimeError(
                        "NR_RAPL guided crop is enabled, but the training "
                        "batch has neither 'guided_crops' nor one image path "
                        "per sample. Check the dataset/crop wrapper configuration."
                    )
                image_size = int(images.shape[-1])
                batch["guided_crops"] = torch.stack(
                    [
                        make_guided_crop_candidates(
                            path,
                            model.nr_config.CROP,
                            image_size=image_size,
                        )
                        for path in paths
                    ],
                    dim=0,
                )
                if i == 0:
                    lib.LOGGER.warning(
                        "'guided_crops' was absent from the DataLoader batch; "
                        "generated CPU crop candidates from batch['path']. "
                        "Synchronize GuidedCropDataset to avoid this fallback."
                    )
            # Candidate tensors stay on host memory. NRRAPL transfers and
            # encodes them in small chunks under torch.no_grad().
            guided_selection = model.select_guided_crops(batch["guided_crops"], targets, class_names)
        with autocast(enabled=args.use_fp16):
            regional_positive_text_features = None
            regional_negative_text_features = None
            if isinstance(model, NRRAPL) and model.uses_regional_contrastive:
                # Encode each learnable regional prompt set once per iteration
                # and reuse the live graph for classifier, L_pos, L_neg, L_div.
                # Do not cache beyond this iteration: optimizer.step() updates
                # these prompts and a detached cache would be incorrect.
                text_features, local_text_features = model.encode_text(class_names)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                local_text_features = local_text_features / local_text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                regional_positive_text_features = local_text_features
                regional_negative_text_features = model.encode_local_negative_prompts()
                if regional_negative_text_features is None:
                    raise RuntimeError("Regional contrastive training requires at least one enabled negative prompt")
            elif isinstance(model, NRRAPL) and model.nr_config.ABLATION.USE_DIV_LOSS:
                regional_negative_text_features = model.encode_local_negative_prompts()

            # Crop-only ablation B has no regional objective. In that precise
            # setting, train RAPL's unchanged classifier on the selected
            # positive crop. Full regional variants retain the original image
            # for RAPL's global/local classification losses.
            model_images = images
            if guided_selection is not None and not model.uses_regional_contrastive:
                model_images = guided_selection["positive_images"].to(images.device, non_blocking=True)
            global_logits, local_logits = model(model_images, class_names, text_features, local_text_features)
            loss_components = loss_fn.compute_components(global_logits, local_logits, targets, model.logit_scale.exp())
            if isinstance(model, NRRAPL):
                loss_global = loss_components["global"]
                loss_local_original = loss_components["local"]
                loss = loss_global
                if model.nr_config.LOSS.KEEP_ORIGINAL_LOCAL_LOSS:
                    loss = loss + model.nr_config.LOSS.LAMBDA_LOCAL_ORIGINAL * loss_local_original
            else:
                loss = loss_components["global"] + loss_components["local"]
                loss_global = loss_components["global"]
                loss_local_original = loss_components["local"]

            loss_neg_div = None
            regional_metrics = None
            lambda_neg_current = 0.0
            if isinstance(model, NRRAPL):
                if model.uses_regional_contrastive:
                    # A positive crop is always selected.  The hard-negative
                    # loss itself masks filtered samples through negative_valid.
                    negative_images = guided_selection["negative_images"] if guided_selection is not None else images
                    negative_valid = guided_selection["negative_valid"] if guided_selection is not None else None
                    if guided_selection is not None and "negative_images_all" in guided_selection:
                        # Aggregate each selected hard negative as an
                        # independent regional sample. Invalid/padded slots
                        # are safely ignored by the negative loss mask.
                        negative_images = guided_selection["negative_images_all"].flatten(0, 1)
                        negative_valid = guided_selection["negative_valid_all"].flatten()
                    regional_metrics = model.compute_regional_contrastive_losses(
                        positive_images=guided_selection["positive_images"] if guided_selection is not None else images,
                        negative_images=negative_images,
                        targets=targets,
                        class_names=class_names,
                        negative_valid=negative_valid,
                        positive_text_features=regional_positive_text_features,
                        negative_text_features=regional_negative_text_features,
                    )
                    lambda_neg_current = model.negative_loss_weight(epoch)
                    loss = loss + model.nr_config.LOSS.LAMBDA_POS * regional_metrics["loss_pos"]
                    loss = loss + lambda_neg_current * regional_metrics["loss_neg"]
                loss_neg_div = model.compute_negative_prompt_diversity(
                    reference=loss,
                    negative_text_features=regional_negative_text_features,
                )
                loss = loss + model.nr_config.LOSS.LAMBDA_DIV * loss_neg_div

        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite loss at epoch={epoch}, batch={i}: {loss.detach().float().item()}")
        fp16_scaler.scale(loss).backward()
        if isinstance(model, NRRAPL):
            gradient_status = model.assert_frozen_gradient_invariants()
            if i == 0:
                lib.LOGGER.info("NR_RAPL gradient status: " + ", ".join(f"{name}={value}" for name, value in gradient_status.items()))
        track_loader.set_postfix({"gpu": torch.cuda.max_memory_allocated() / 1024**3})
        fp16_scaler.step(optimizer)
        fp16_scaler.update()
        optimizer.zero_grad()

        gl_probs, global_probs, local_probs = model.create_prediction_scores(global_logits, local_logits)
        topk = accuracy(gl_probs, targets, topk=(1,))
        global_topk = accuracy(global_probs, targets, topk=(1,))

        logs = {
                "loss": loss.detach().item(),
                "top1": topk[0],
                "top1_global": global_topk[0],
        }
        if isinstance(model, NRRAPL):
            logs.update(
                {
                    "total_loss": loss.detach().item(),
                    "loss_global": loss_global.detach().item(),
                    "loss_local_original": loss_local_original.detach().item(),
                    "loss_neg_div": loss_neg_div.detach().item(),
                }
            )
            if regional_metrics is not None:
                logs.update(
                    {
                        "loss_pos": regional_metrics["loss_pos"].detach().item(),
                        "loss_neg": regional_metrics["loss_neg"].detach().item(),
                        "weighted_loss_neg": (lambda_neg_current * regional_metrics["loss_neg"]).detach().item(),
                        "lambda_neg": lambda_neg_current,
                        "positive_regional_accuracy": regional_metrics["positive_regional_accuracy"].detach().item(),
                        "negative_prompt_probability": regional_metrics["negative_prompt_probability"].detach().item(),
                        "valid_negative_ratio": regional_metrics["valid_negative_ratio"].detach().item(),
                    }
                )

        if guided_selection is not None:
            margins = guided_selection["candidate_margins"]
            positive_indices = guided_selection["positive_indices"].unsqueeze(1)
            negative_indices = guided_selection["negative_indices"].unsqueeze(1)
            positive_margin = margins.gather(1, positive_indices).squeeze(1)
            negative_margin = margins.gather(1, negative_indices).squeeze(1)
            logs.update(
                {
                    "crop_margin_pos": positive_margin.mean().item(),
                    "crop_margin_neg": negative_margin.mean().item(),
                    "crop_margin_gap": (positive_margin - negative_margin).mean().item(),
                    "crop_negative_valid": guided_selection["negative_valid"].float().mean().item(),
                    "crop_repeated": guided_selection["negative_repeated"].float().mean().item(),
                }
            )

        meter.update(logs, images.size(0))

        if local_probs is not None:
            local_topk = accuracy(local_probs, targets, topk=(1,))
            meter.update(
                {
                    "top1_local": local_topk[0],
                },
                images.size(0),
            )

    progress.display_summary()

    lr_scheduler.step()
    return meter


@torch.no_grad()
def evaluate(
    model: RAPL,
    val_loader: DataLoader,
    args: argparse.Namespace,
    return_scores: bool = False,
) -> Tuple[lib.DictAverage, np.ndarray]:
    meter = lib.DictAverage()

    class_names = val_loader.dataset.all_names

    with autocast(enabled=args.use_fp16):
        text_features, local_text_features = model.encode_text(class_names)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)

    mode = model.training
    model.eval()
    test_scores = np.zeros(len(val_loader.dataset))
    dataset_name = val_loader.dataset.__class__.__name__[:-7]
    for batch in lib.track(val_loader, f"Evaluating on {dataset_name}"):
        images = batch["image"].cuda(non_blocking=True)
        targets = batch["target"].cuda(non_blocking=True)

        with autocast(enabled=args.use_fp16):
            global_logits, local_logits = model(images, text_features=text_features, local_text_features=local_text_features)

            if return_scores:
                if isinstance(model, NRRAPL) and model.uses_id_confidence_ood_scorer:
                    # Library OOD metrics label OOD as positive, while NR-RAPL
                    # exposes a high-ID-confidence final score.
                    details = model.score_ood_images(images, class_names, global_logits, local_logits)
                    test_scores[batch["index"].numpy()] = -details["final_ood_score"].cpu().numpy()
                else:
                    test_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)

        gl_probs, global_probs, local_probs = model.create_prediction_scores(global_logits, local_logits)
        global_topk = accuracy(global_probs, targets, topk=(1,))

        if local_probs is not None:
            local_topk = accuracy(local_probs, targets, topk=(1,))

            topk = accuracy(gl_probs, targets, topk=(1,))

            logs = {
                "top1": topk[0],
                "top1_global": global_topk[0],
                "top1_local": local_topk[0],
            }
        else:
            logs = {
                "top1": global_topk[0],
                "top1_global": global_topk[0],
            }

        meter.update(logs, images.size(0))

    model.train(mode)
    return meter, test_scores


@torch.no_grad()
def evaluate_ood(
    model: RAPL,
    val_loader: DataLoader,
    ood_loaders: Dict[str, DataLoader],
    args: argparse.Namespace,
    test_scores: Optional[np.ndarray] = None,
) -> lib.DictAverage:
    metrics = defaultdict(dict)

    class_names = val_loader.dataset.all_names

    with autocast(enabled=args.use_fp16):
        text_features, local_text_features = model.encode_text(class_names)
        text_features /= text_features.norm(dim=-1, keepdim=True)
        local_text_features /= local_text_features.norm(dim=-1, keepdim=True)

    mode = model.training
    model.eval()
    if test_scores is None:
        test_scores = np.zeros(len(val_loader.dataset))
        for batch in lib.track(val_loader, "Computing ood scores for Test"):
            images = batch["image"].cuda(non_blocking=True)
            with autocast(enabled=args.use_fp16):
                global_logits, local_logits = model(images, text_features=text_features, local_text_features=local_text_features)
                if isinstance(model, NRRAPL) and model.uses_id_confidence_ood_scorer:
                    details = model.score_ood_images(images, class_names, global_logits, local_logits)
                    test_scores[batch["index"].numpy()] = -details["final_ood_score"].cpu().numpy()
                else:
                    test_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)

    for ood_name, ood_loader in ood_loaders.items():
        ood_scores = np.zeros(len(ood_loader.dataset))
        for batch in lib.track(ood_loader, f"Computing ood scores for {ood_name}"):
            images = batch["image"].cuda(non_blocking=True)
            with autocast(args.use_fp16):
                global_logits, local_logits = model(images, text_features=text_features, local_text_features=local_text_features)
                if isinstance(model, NRRAPL) and model.uses_id_confidence_ood_scorer:
                    details = model.score_ood_images(images, class_names, global_logits, local_logits)
                    ood_scores[batch["index"].numpy()] = -details["final_ood_score"].cpu().numpy()
                else:
                    ood_scores[batch["index"].numpy()] = model.compute_scores(global_logits, local_logits)

        metrics[ood_name]["fpr95"] = lib.get_fpr(test_scores, ood_scores)
        metrics[ood_name]["auroc"] = lib.get_auroc(test_scores, ood_scores)

    model.train(mode)
    return metrics


if __name__ == "__main__":
    clip_model_names = [
        "clip_vit_b32",
        "clip_vit_b16",
        "clip_resnet50",
        "clip_resnet101",
    ]

    parser = argparse.ArgumentParser("Learning prompts for CLIP with local and global features")
    parser.add_argument("--exp_name", default=None, type=str)
    parser.add_argument("--data_dir", default="/share/DEEPLEARNING/datasets", type=str)
    parser.add_argument("--save_dir", default="./results/", type=str)
    parser.add_argument("--ablation_name", default=None, type=str)
    parser.add_argument("--results_csv", default=None, type=str)
    parser.add_argument("--checkpoint_path", default=None, type=str)
    parser.add_argument("--checkpoint_name", default=None, type=str)
    parser.add_argument("--dataset_name", default="imagenet", type=str)
    parser.add_argument("--eval_only", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_ood", default=False, type=lib.boolean_flags)
    parser.add_argument("--eval_domains", default=False, type=lib.boolean_flags)

    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--num_shots", default=16, type=int, help="Number of shots by class. -1 means the whole dataset")
    parser.add_argument("--use_local_features", default=False, type=lib.boolean_flags)
    parser.add_argument("--use_global_loss", default=False, type=lib.boolean_flags)
    parser.add_argument("--use_local_loss", default=True, type=lib.boolean_flags)
    parser.add_argument("--topk", default=[5, 10, 15, 20], type=int, nargs="+")
    parser.add_argument("--learn_local_proj", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_global_prompt", default=True, type=lib.boolean_flags)
    parser.add_argument("--learn_local_prompts", default=True, type=lib.boolean_flags)
    parser.add_argument("--n_global_prompts", default=1, type=int)
    parser.add_argument("--n_local_prompts", default=1, type=int)
    parser.add_argument("--global_dropout_p", default=0.75, type=lib.float_range(0.0, 1.0))

    parser.add_argument("--prompts_batch_size", default=math.inf, type=int)

    parser.add_argument("--parallel_text_encoder", default=False, type=lib.boolean_flags)
    parser.add_argument("--parallel_vision_encoder", default=False, type=lib.boolean_flags)

    parser.add_argument("--ood_method", default="GL-MCM", type=str)
    parser.add_argument("--ood_temp_scale", default=1.0, type=float)

    parser.add_argument("--clip_name", required=True, choices=clip_model_names, type=str)

    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--inference_batch_size", default=256, type=int)
    parser.add_argument("--max_epoch", default=50, type=int)
    parser.add_argument("--optimizer", default="sgd", type=str)
    parser.add_argument("--lr_init", default=0.002, type=float)
    parser.add_argument("--momentum", default=0.9, type=float)
    parser.add_argument("--weight_decay", default=1e-2, type=float)
    parser.add_argument("--warmup_epoch", default=0, type=int)
    parser.add_argument("--cons_lr", default=1e-5, type=float)

    parser.add_argument("--use_fp16", default=True, type=lib.boolean_flags)
    parser.add_argument("--persistent_workers", default=False, type=lib.boolean_flags)
    parser.add_argument("--checkpointing_segments", default=4, type=int, help="Number of segments used for gradient checkpointing for the text encoder.")

    parser.add_argument("--eval_freq", default=5, type=int)
    parser.add_argument("--save_freq", default=5, type=int)
    parser.add_argument("--print_freq", default=20, type=int)

    # NR_RAPL is opt-in.  Defaults retain the original RAPL code path.
    add_NR_RAPL_arguments(parser, lib.boolean_flags)

    args = parser.parse_args()
    method_config = build_method_config(args)

    lib.setup_logger()
    lib.random_seed(args.seed)

    if args.exp_name is not None:
        lib.LOGGER.info(f"Running experiment {args.exp_name}")
        args.save_dir = os.path.join(args.save_dir, args.exp_name)

    args.eval_domains = args.eval_domains and (args.dataset_name == "imagenet")
    args.eval_ood = args.eval_ood and (args.dataset_name == "imagenet")

    # Accept both the project data root (.../DATA) and the ImageNet root
    # (.../DATA/Imagenet). Dataset/OOD helpers consistently expect the former.
    if args.dataset_name == "imagenet":
        imagenet_train = os.path.join(args.data_dir, "train")
        imagenet_val = os.path.join(args.data_dir, "val")
        if os.path.isdir(imagenet_train) and os.path.isdir(imagenet_val):
            args.data_dir = os.path.dirname(os.path.normpath(args.data_dir))
            lib.LOGGER.info(f"Detected ImageNet root; using project data root {args.data_dir}")

    # Save the resolved arguments, including dataset-dependent evaluation
    # switches, rather than only the raw command-line request.
    reporter = ExperimentReporter(args, method_config)
    reporter.save_configuration()
    reporter.log_modules(lib.LOGGER)

    # seting-up transforms
    train_transform = dts_tools.get_train_transform()
    val_transform = _transform(224)

    # Setting-up Imagenet dataset train
    train_dataset, val_dataset, template = return_train_val_datasets(args.dataset_name, args.data_dir, train_transform, val_transform)
    template = "A photo of a {}" if (args.learn_global_prompt or args.learn_local_prompts) else template

    train_dataset = dts_tools.create_few_shots_dataset(train_dataset, args.num_shots, seed=args.seed)
    # Use the command-line switches here as well as in ``train_one_epoch``.
    # This prevents a stale/deserialised method config from building a plain
    # dataset while the model subsequently expects ``batch['guided_crops']``.
    use_guided_crop_dataset = (
        args.method_name == "NR_RAPL"
        and args.nr_crop_enabled
        and args.nr_use_guided_crop
    )
    if use_guided_crop_dataset:
        # Wrap before DataLoader construction.  The wrapper attaches a CPU
        # tensor named ``guided_crops`` to every training sample, which the
        # NR selector later encodes in bounded GPU chunks.
        train_dataset = maybe_wrap_guided_crop_dataset(
            train_dataset,
            method_config.NR_RAPL.CROP,
            use_guided_crop=True,
        )
        if not isinstance(train_dataset, GuidedCropDataset):
            raise RuntimeError(
                "NR_RAPL guided crop was requested, but the training dataset "
                "was not wrapped as GuidedCropDataset. Verify "
                "RAPL.datasets.tools.guided_crops is present on the Python path."
            )
        lib.LOGGER.info(
            "NR_RAPL guided-crop dataset enabled: "
            f"{method_config.NR_RAPL.CROP.NUM_CANDIDATES} candidates/sample "
            f"(dataset={type(train_dataset).__name__})"
        )
    lib.LOGGER.info("Using template: " + template.format("<class_name>"))

    # Setting-up dataloaders
    train_loader = dts_tools.get_train_loader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=10,
        persistent_workers=args.persistent_workers,
    )
    val_loader = dts_tools.get_eval_loader(val_dataset, batch_size=args.inference_batch_size)

    if args.eval_ood:
        ood_loaders = return_ood_loaders(args.data_dir, val_transform)

    if args.eval_domains:
        domains_loaders = return_domains_loaders(args.data_dir, val_transform)

    # Setting-up model
    if method_config.METHOD.NAME == "RAPL":
        # Keep the original constructor invocation and its defaults intact.
        model = RAPL(
            clip_name=args.clip_name,
            use_local_features=args.use_local_features,
            checkpointing_segments=args.checkpointing_segments,
            template=template,
            learn_local_proj=args.learn_local_proj,
            learn_local_prompts=args.learn_local_prompts,
            learn_global_prompt=args.learn_global_prompt,
            class_names=train_dataset.all_names,
            n_global_prompts=args.n_global_prompts,
            n_local_prompts=args.n_local_prompts,
            prompts_batch_size=args.prompts_batch_size,
            ood_method=args.ood_method,
            ood_temp_scale=args.ood_temp_scale,
            topk=args.topk,
            parallel_text_encoder=args.parallel_text_encoder,
            parallel_vision_encoder=args.parallel_vision_encoder,
        )
    else:
        model = NRRAPL(
            nr_config=method_config.NR_RAPL,
            clip_name=args.clip_name,
            use_local_features=args.use_local_features,
            checkpointing_segments=args.checkpointing_segments,
            template=template,
            learn_local_proj=args.learn_local_proj,
            learn_local_prompts=args.learn_local_prompts,
            learn_global_prompt=args.learn_global_prompt,
            class_names=train_dataset.all_names,
            n_global_prompts=args.n_global_prompts,
            n_local_prompts=args.n_local_prompts,
            prompts_batch_size=args.prompts_batch_size,
            ood_method=args.ood_method,
            ood_temp_scale=args.ood_temp_scale,
            topk=args.topk,
            parallel_text_encoder=args.parallel_text_encoder,
            parallel_vision_encoder=args.parallel_vision_encoder,
        )

    model.initialize_prompt()
    if isinstance(model, NRRAPL):
        model.refresh_frozen_global_guide(force=True)

    # eventually load pre-trained prompts
    lib.load_checkpoint(model, args.checkpoint_path)
    if isinstance(model, NRRAPL):
        model.refresh_frozen_global_guide()

    model.freeze_clip()
    if isinstance(model, NRRAPL):
        model.configure_train_stage()
    model = model.cuda()
    torch.cuda.reset_peak_memory_stats()

    # setting-up loss
    loss_fn = GlobalLocalLoss(
        use_global_loss=args.use_global_loss,
        use_local_loss=args.use_local_loss,
        topk=args.topk,
        global_dropout_p=args.global_dropout_p,
    )

    # Stage 2/3 use explicit NR groups; legacy RAPL stays on its original
    # optimizer construction path.
    if isinstance(model, NRRAPL):
        optimizer = get_NR_RAPL_optimizer(args.optimizer, model, args.lr_init, args.weight_decay, args.momentum)
    else:
        optimizer = vlp_tools.get_optimizer(args.optimizer, model, args.lr_init, args.weight_decay, args.momentum)

    # Setting-up scheduler
    lr_scheduler = CosineAnnealingLR(optimizer, args.max_epoch)
    if args.warmup_epoch > 0:
        lr_scheduler = vlp_tools.ConstantWarmupScheduler(optimizer, lr_scheduler, args.warmup_epoch, args.cons_lr)

    # Setting-up GradScaler for amp (PyTorch 1.x and 2.6+ compatible).
    fp16_scaler = _create_grad_scaler(enabled=args.use_fp16)

    start_epoch = 0
    if isinstance(model, NRRAPL) and model.nr_config.RESUME_OPTIMIZER:
        restored_epoch = model.restore_training_state(optimizer, lr_scheduler, fp16_scaler)
        if restored_epoch is not None:
            start_epoch = restored_epoch + 1
            lib.LOGGER.info(f"Resumed NR_RAPL training from epoch {start_epoch}")

    best_metric = float("-inf")
    best_epoch = None
    train_seconds = 0.0
    test_seconds = 0.0
    last_id_accuracy = None
    ood_metrics = {}
    # Training loop
    for epoch in range(start_epoch, args.max_epoch):
        if not args.eval_only:
            assert args.use_local_loss or args.use_global_loss or args.learn_local_prompts or args.learn_global_prompt, "At least one of use_local_loss or use_global_loss or learn_local_prompts or learn_global_prompt must be True"
            train_start = time.perf_counter()
            train_meter = train_one_epoch(
                model=model,
                train_loader=train_loader,
                loss_fn=loss_fn,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                epoch=epoch,
                fp16_scaler=fp16_scaler,
                args=args,
            )
            train_seconds += time.perf_counter() - train_start

            if not isinstance(model, NRRAPL) and args.checkpoint_name is None:
                lib.save_checkpoint(
                    args.save_dir,
                    epoch,
                    model,
                    optimizer,
                    lr_scheduler,
                    fp16_scaler,
                    train_meter,
                    args,
                    checkpoint_name=args.checkpoint_name,
                )

        if ((epoch % args.eval_freq == 0) and (epoch > 0)) or (epoch + 1 == args.max_epoch) or args.eval_only:
            lib.LOGGER.info("Evaluation")
            test_start = time.perf_counter()
            val_meter, test_scores = evaluate(model, val_loader, args, return_scores=args.eval_ood and (args.eval_only or (epoch + 1 == args.max_epoch)))
            test_seconds += time.perf_counter() - test_start
            last_id_accuracy = val_meter.avg.get("top1")
            lib.LOGGER.info("Evaluation metrics: " + " ".join([" *"] + val_meter.summary()))

            if (isinstance(model, NRRAPL) or args.checkpoint_name is not None) and not args.eval_only:
                metric = val_meter.avg.get("top1", float("-inf"))
                if metric >= best_metric:
                    best_metric = metric
                    best_epoch = epoch
                    lib.save_checkpoint(
                        args.save_dir,
                        epoch,
                        model,
                        optimizer,
                        lr_scheduler,
                        fp16_scaler,
                        train_meter,
                        args,
                        best_metric=best_metric,
                        checkpoint_name=args.checkpoint_name,
                    )
                    lib.LOGGER.info(f"Saved NR_RAPL best checkpoint (ID top1={best_metric:.4f})")

            if args.eval_ood and (args.eval_only or (epoch + 1 == args.max_epoch)):
                test_start = time.perf_counter()
                ood_metrics = evaluate_ood(model, val_loader, ood_loaders, args, test_scores=test_scores)
                test_seconds += time.perf_counter() - test_start
                lib.LOGGER.info(f"OOD Evaluation metrics with temperature scale {args.ood_temp_scale} (FPR95 / AUROC): ")
                lib.log_ood_metrics(ood_metrics)

            if args.eval_domains and (args.eval_only or (epoch + 1 == args.max_epoch)):
                metrics = {}
                for domain_name, domain_loader in domains_loaders.items():
                    metrics[domain_name], _ = evaluate(model, domain_loader, args)
                    lib.LOGGER.info(f"Evaluation metrics for {domain_name}: " + " ".join([" *"] + metrics[domain_name].summary()))
                avg_top1 = np.mean([metrics[domain_name].avg["top1"] for domain_name in domains_loaders.keys()])
                lib.LOGGER.info(f"Average evaluation metrics for domains: * top1: {avg_top1: .3f}")

        if args.eval_only:
            break

    checkpoint_path = None
    if args.checkpoint_name is not None:
        checkpoint_path = os.path.join(args.save_dir, args.checkpoint_name)
    reporter.append_result(
        id_accuracy=last_id_accuracy,
        ood_metrics=ood_metrics,
        best_epoch=best_epoch,
        train_seconds=train_seconds,
        test_seconds=test_seconds,
        max_memory_gb=torch.cuda.max_memory_allocated() / 1024**3,
        checkpoint_path=checkpoint_path,
    )
    lib.LOGGER.info(f"Ablation result appended to {reporter.csv_path}")
