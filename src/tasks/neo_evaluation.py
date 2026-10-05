"""NEO evaluation: select on ID query, then freeze and test.

Task-specific inference lives in each task's evaluator. This module owns
checkpoint selection, artifact provenance and stage ordering.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path

import h5py
import torch

from tasks.checkpoint_selection import (
    HARD_GROUNDING, build_model, checkpoint_hash, choose_best, load_candidate, read_episodes, score_id,
)
from training.runtime import (
    ExperimentTrackingConfig, MetricLog, capture_git_source, seed_everything,
)

TASK_DOMAINS = {"gridworld": "gridworld", "image_editing": "image",
                "arithmetic_factorization": "arithmetic"}
ALPHAS = ("0.33", "0.66", "1.00")
SEEDS = (42, 43, 44)
SELECTION_POLICY = "task-grounding-id-query-v2"
DEVICE = "cuda"
ARITHMETIC_EPISODES = 5000


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def data_path(domain: str, alpha: str, split: str, root: Path) -> Path:
    if domain == "gridworld":
        from tasks.gridworld.theorizer_evaluation import evaluation_artifact
        return root / evaluation_artifact(f"alpha-{alpha}", split).filename
    if domain == "image":
        from tasks.image_editing.evaluation import resolve_protocol_h5
        return resolve_protocol_h5(alpha, split.replace("compositional", "comp"), root)
    from tasks.arithmetic_factorization.data.profiles import profile_for_alpha
    profile = profile_for_alpha(alpha)
    artifact = {"id": profile.test, "compositional-ood": profile.composition_ood,
                "length-ood": profile.length_ood}[split]
    if artifact is None:
        raise ValueError(f"no {split} data for alpha {alpha}")
    return root / artifact.filename


def data_identity(path: Path) -> dict:
    with h5py.File(path, "r") as data:
        count = int(data.attrs["dataset_length"])
    return {"path": str(path.resolve()), "sha256": checkpoint_hash(path),
            "bytes": path.stat().st_size, "episodes": count}


def checkpoint_directory(domain: str, alpha: str, seed: int, root: Path) -> Path:
    if domain == "gridworld":
        name = f"gridworld-alpha-{alpha}-seed-{seed}"
    elif domain == "image":
        name = f"image-editing-neo-alpha{alpha}-seed{seed}"
    else:
        from tasks.arithmetic_factorization.theorizer_runner import run_name
        name = run_name("neo", alpha, seed)
    directories = sorted((root / name).glob("*/checkpoints"))
    if len(directories) != 1:
        raise ValueError(f"expected one run under {root / name}; found {len(directories)}. "
                         "Use --checkpoint-directory to choose one run explicitly.")
    return directories[0]


def select_checkpoint(*, domain: str, alpha: str, seed: int, directory: Path,
                      id_path: Path, output: Path, device: str, tracking: dict,
                      source: dict) -> dict:
    """Finish the full ID-only sweep before publishing a selected checkpoint.

    A completed selection is reusable only with identical code, inputs and
    candidate hashes. A partial/failed sweep is preserved; use a new output root.
    """
    paths = sorted(set(directory.glob("*.pt")) | set(directory.glob("*.pth")))
    if not paths:
        raise FileNotFoundError(f"no saved checkpoints in {directory}")
    identity = data_identity(id_path)
    inventory = [{"checkpoint": str(p.resolve()), "checkpoint_sha256": checkpoint_hash(p)}
                 for p in paths]
    contract = {"policy": SELECTION_POLICY, "domain": domain, "alpha": alpha, "seed": seed,
                "id_data": identity, "candidates": inventory, "source": source,
                "hard_grounding": HARD_GROUNDING[domain]}
    selected_path = output / "selected.json"
    if selected_path.exists():
        previous = json.loads(selected_path.read_text())
        if previous["contract"] != contract:
            raise ValueError("selection inputs/code changed; use a new --output-root")
        return previous
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "manifest.json", contract)
    seed_everything(seed)
    episodes = read_episodes(id_path, domain)
    model = build_model(domain, alpha, device)
    log = MetricLog(output / "metrics.jsonl",
                    ExperimentTrackingConfig(name=f"{domain}-{alpha}-{seed}-selection", **tracking),
                    contract)
    success = False
    try:
        records = []
        for index, item in enumerate(inventory):
            checkpoint = Path(item["checkpoint"])
            if checkpoint_hash(checkpoint) != item["checkpoint_sha256"]:
                raise ValueError(f"checkpoint changed during selection: {checkpoint}")
            state = load_candidate(model, domain, alpha, seed, checkpoint)
            score = score_id(model, domain, episodes, device=device, coefficient=state["coefficient"])
            record = {**item, **state, **score}
            records.append(record)
            write_json(output / f"candidate-{index:03d}.json", record)
            log.log({"candidate": index, "id_query": score["score"], "global_step": state["global_step"]})
            write_json(output / "progress.json", {"completed": len(records), "total": len(inventory)})
        best = choose_best(records)
        result = {"checkpoint": best["checkpoint"], "checkpoint_sha256": best["checkpoint_sha256"],
                  "selection": best, "candidate_count": len(records), "id_data": identity,
                  "id_is_selection_data": True, "contract": contract}
        write_json(selected_path, result)
        success = True
        return result
    finally:
        log.finish(exit_code=0 if success else 1)


def read_selection(path: Path, *, domain: str, alpha: str, seed: int, id_path: Path,
                   device: str) -> dict:
    """Import a frozen ID-only selection written by :func:`select_checkpoint`."""
    record = json.loads(path.read_text())
    checkpoint = Path(record["checkpoint"]).expanduser()
    if not checkpoint.is_absolute():
        checkpoint = path.resolve().parent / checkpoint
    record = {**record, "checkpoint": str(checkpoint.resolve())}
    chosen = {**record["selection"], "checkpoint": record["checkpoint"]}
    record["selection"] = chosen
    choose_best([chosen])
    if chosen["hard_grounding"] != HARD_GROUNDING[domain]:
        raise ValueError("selection grounding setting differs from the task evaluation setting")
    identity = data_identity(id_path)
    if any(record["id_data"][key] != identity[key] for key in ("sha256", "episodes")):
        raise ValueError("selection ID artifact does not match this task/alpha")
    if chosen["episodes"] != identity["episodes"]:
        raise ValueError("selection did not use the full ID artifact")
    if record["checkpoint_sha256"] != checkpoint_hash(Path(record["checkpoint"])):
        raise ValueError("selected checkpoint hash mismatch")
    if chosen["checkpoint_sha256"] != record["checkpoint_sha256"]:
        raise ValueError("selection score and checkpoint identify different weights")
    contract = record["contract"]
    if (contract["domain"], contract["alpha"], contract["seed"]) != (domain, alpha, seed):
        raise ValueError("selection belongs to a different model condition")
    model = build_model(domain, alpha, device)
    load_candidate(model, domain, alpha, seed, Path(record["checkpoint"]))
    return {**record, "imported_selection": str(path.resolve())}


def evaluate_arithmetic(checkpoint: Path, path: Path, *, alpha: str, seed: int,
                        split: str, device: str, scaling: bool) -> dict:
    """Greedy FP32 rollout on the first 5,000 episodes, optionally with test-time scaling."""
    from tasks.arithmetic_factorization.data.dataset import build_dataloader, unpack_batch
    from tasks.arithmetic_factorization.latent_inference import latent_candidates
    from tasks.arithmetic_factorization.theorizer_scaling import (
        evaluate_test_time_scaling, PAPER_SCALING_BUDGETS,
    )
    model = build_model("arithmetic", alpha, device)
    load_candidate(model, "arithmetic", alpha, seed, checkpoint)
    loader = build_dataloader(path, batch_size=128, shuffle=False, num_workers=0)
    horizon = 6 if split == "length-ood" else 3
    n = correct = support = filtered = 0
    for batch in loader:
        episodes, _ = unpack_batch(batch, device)
        for episode in episodes:
            result = latent_candidates(model, episode, max_steps=horizon, greedy=True, hard_grounding=True)
            correct += int(result.query_correct.item())
            support += int(result.support_correct.item())
            filtered += int((result.query_correct & result.support_correct).item())
            n += 1
            if n == ARITHMETIC_EPISODES:
                break
        if n == ARITHMETIC_EPISODES:
            break
    if n != ARITHMETIC_EPISODES:
        raise ValueError(f"expected {ARITHMETIC_EPISODES} Arithmetic episodes, found {n}")
    metrics = {"greedy": {"episodes": n, "transfer": correct / n, "support": support / n,
                          "support_filtered_transfer": filtered / n},
               "coefficient": model.length_control_coefficient, "precision": "float32"}
    if scaling:
        torch.manual_seed(42)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(42)
        metrics["scaling"] = evaluate_test_time_scaling(
            model, loader, max_steps=horizon, budgets=PAPER_SCALING_BUDGETS, sample_temperature=1.0,
            device=device, num_samples=ARITHMETIC_EPISODES, hard_grounding=True).legacy_dict()
        metrics["sampling_rng_seed"] = 42
    return metrics


def evaluate_selected(*, domain: str, alpha: str, seed: int, selected: dict,
                      data_root: Path, output: Path, protocol: str, device: str,
                      tracking: dict, source: dict) -> list[dict]:
    """Only this stage resolves/opens OOD inputs; the winner is already frozen."""
    hard_grounding = HARD_GROUNDING[domain]
    if selected["selection"]["hard_grounding"] != hard_grounding:
        raise ValueError("selection and evaluation must use the same grounding setting")
    checkpoint = Path(selected["checkpoint"])
    if checkpoint_hash(checkpoint) != selected["checkpoint_sha256"]:
        raise ValueError("selected checkpoint changed before evaluation")
    output.mkdir(parents=True, exist_ok=False)
    records = []
    splits = ("id", "length-ood") if alpha == "1.00" else ("id", "compositional-ood", "length-ood")
    for split in splits:
        seed_everything(seed)
        path = data_path(domain, alpha, split, data_root)
        identity = data_identity(path)
        config = {"domain": domain, "alpha": alpha, "seed": seed, "split": split,
                  "protocol": protocol, "hard_grounding": hard_grounding, "id_is_selection_data": True,
                  "selected": selected, "data": identity, "source": source,
                  "torch": torch.__version__, "cuda": torch.version.cuda,
                  "device": device, "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
        trackdir = output / split
        trackdir.mkdir()
        log = MetricLog(trackdir / "metrics.jsonl",
                        ExperimentTrackingConfig(name=f"{domain}-{alpha}-{seed}-{split}-{protocol}", **tracking), config)
        success = False
        try:
            if domain == "arithmetic":
                metrics = evaluate_arithmetic(checkpoint, path, alpha=alpha, seed=seed, split=split,
                                              device=device, scaling=protocol == "test-time-scaling")
            elif domain == "image":
                from tasks.image_editing.evaluation import EvaluationRunConfig, evaluate_checkpoint
                metrics = evaluate_checkpoint(EvaluationRunConfig(
                    alpha=alpha, protocol=split.replace("compositional", "comp"),
                    checkpoint=checkpoint, data_h5=path, device=device, seed=seed,
                    hard_grounding=hard_grounding))
            else:
                from tasks.gridworld.theorizer_evaluation import GridWorldEvaluationRunConfig, run_theorizer_evaluation
                # The GridWorld evaluator checks its artifacts and tracks the run itself.
                log.finish(exit_code=0)
                log = None
                run, result = run_theorizer_evaluation(GridWorldEvaluationRunConfig(
                    experiment=f"alpha-{alpha}", model_seed=seed, split=split, checkpoint=checkpoint,
                    data_h5=path, output_root=trackdir / "evaluation", protocol=protocol,
                    hard_grounding=hard_grounding,
                    wandb_project=tracking["project"], wandb_entity=tracking.get("entity"),
                    wandb_group=tracking.get("group"), wandb_mode=tracking["mode"]))
                metrics = {protocol: result.legacy_dict(), "output": str(run)}
            record = {**config, "metrics": metrics}
            write_json(output / f"{split}.json", record)
            if log:
                log.log({"completed": 1})
            records.append(record)
            success = True
        finally:
            if log:
                log.finish(exit_code=0 if success else 1)
    return records


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=TASK_DOMAINS, required=True)
    parser.add_argument("--alpha", choices=(*ALPHAS, "all"), default="all")
    parser.add_argument("--seed", choices=("42", "43", "44", "all"), required=True)
    parser.add_argument("--training-root", type=Path)
    parser.add_argument("--checkpoint-directory", type=Path)
    parser.add_argument("--selection-record", type=Path)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--protocol", choices=("standard", "test-time-scaling"), default="standard")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--wandb-project", default="LearningToTheorize")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--wandb-group")
    parser.add_argument("--wandb-mode", choices=("online", "offline"), default="online")
    args = parser.parse_args(argv)
    domain = TASK_DOMAINS[args.task]
    if domain == "image" and args.protocol != "standard":
        parser.error("ImageEditing has no NEO-S protocol")
    if args.selection_record and args.checkpoint_directory:
        parser.error("choose --selection-record or --checkpoint-directory")
    if (args.selection_record or args.checkpoint_directory) and (args.alpha == "all" or args.seed == "all"):
        parser.error("an explicit checkpoint directory/selection requires one alpha and seed")
    if not (args.training_root or args.checkpoint_directory or args.selection_record):
        parser.error("provide --training-root, --checkpoint-directory, or --selection-record")
    alphas = ALPHAS if args.alpha == "all" else (args.alpha,)
    seeds = SEEDS if args.seed == "all" else (int(args.seed),)
    if args.dry_run:
        print(json.dumps({"task": args.task, "alphas": alphas, "seeds": seeds,
                          "selection": "full ID query; no OOD", "hard_grounding": HARD_GROUNDING[domain],
                          "protocol": args.protocol}))
        return 0
    torch.set_num_threads(1)
    source = asdict(capture_git_source(Path(__file__).parent))
    tracking = {"project": args.wandb_project, "entity": args.wandb_entity,
                "group": args.wandb_group, "mode": args.wandb_mode}
    for alpha in alphas:
        for seed in seeds:
            key = f"{domain}-alpha{alpha}-seed{seed}"
            id_path = data_path(domain, alpha, "id", args.data_root)
            if args.selection_record:
                selected = read_selection(args.selection_record, domain=domain, alpha=alpha, seed=seed,
                                          id_path=id_path, device=DEVICE)
            else:
                directory = args.checkpoint_directory or checkpoint_directory(domain, alpha, seed, args.training_root)
                selected = select_checkpoint(domain=domain, alpha=alpha, seed=seed, directory=directory,
                    id_path=id_path, output=args.output_root / "selection" / key, device=DEVICE,
                    tracking=tracking, source=source)
            evaluate_selected(domain=domain, alpha=alpha, seed=seed, selected=selected,
                data_root=args.data_root, output=args.output_root / "evaluation" / key / args.protocol,
                protocol=args.protocol, device=DEVICE, tracking=tracking, source=source)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
