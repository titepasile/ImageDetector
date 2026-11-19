"""Helpers for the experiment plan (doc/experiment_plan.md).

python -m scripts.experiments perturb data/genimage_adm/samples.jsonl test_id data/perturbed/jpeg75 jpeg:75
python -m scripts.experiments evaluate outputs/analysis outputs/e3-dual-s17/best.pt --name e3-dual-s17 \
    --test adm256=data/genimage_adm/samples.jsonl:test_id --val
python -m scripts.experiments summarise outputs/analysis
python -m scripts.experiments spectra outputs/analysis/spectra adm=data/genimage_adm/samples.jsonl:test_id
python -m scripts.experiments occlusion outputs/adm-dual/best.pt data/genimage_adm/samples.jsonl test_id outputs/analysis/occlusion/adm-dual
python -m scripts.experiments subset data/genimage_adm_20k/samples.jsonl 2000 data/genimage_adm_20k/samples_2k.jsonl
python -m scripts.experiments merge data/joint/samples.jsonl data/genimage_adm_128/samples.jsonl:2000:500 ...
"""

import argparse
from collections import Counter
import json
from pathlib import Path

import torch


def parse_set(text):
    name, _, rest = text.partition("=")
    manifest, _, split = rest.rpartition(":")
    return name, (manifest, split)


def subset(manifest, per_class, output):
    """Keep the first per_class training images of each class; other splits unchanged."""
    lines = Path(manifest).read_text().splitlines()
    counts, kept = Counter(), []
    for line in lines:
        record = json.loads(line)
        if record["split"] == "train":
            if counts[record["label"]] >= per_class:
                continue
            counts[record["label"]] += 1
        kept.append(line)
    if min(counts[0], counts[1]) < per_class:
        raise ValueError(f"Only {dict(counts)} training images available")
    Path(output).write_text("\n".join(kept) + "\n")
    return dict(counts)


def merge(output, sources):
    """sources: manifest:train_per_class:val_per_class. Paths become absolute.

    Real images are deduplicated by their original file name across all sources,
    so the same ImageNet photo cannot appear twice (or in two splits).
    """
    seen, merged, counts = set(), [], Counter()
    for source in sources:
        manifest, train_count, val_count = source.rsplit(":", 2)
        limits = {"train": int(train_count), "val_id": int(val_count)}
        base = Path(manifest).resolve().parent
        local = Counter()
        for line in Path(manifest).read_text().splitlines():
            record = json.loads(line)
            split = record["split"]
            if split not in limits or local[split, record["label"]] >= limits[split]:
                continue
            original = Path(record.get("source_member", record["path"])).name
            if record["label"] == 0:
                if original in seen:
                    counts["skipped_duplicate_real"] += 1
                    continue
                seen.add(original)
            record["path"] = str((base / record["path"]).resolve())
            local[split, record["label"]] += 1
            counts[f"{Path(manifest).parent.name}:{split}:{record['label']}"] += 1
            merged.append(json.dumps(record))
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    Path(output).write_text("\n".join(merged) + "\n")
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description="Experiment helpers")
    commands = parser.add_subparsers(dest="command", required=True)

    command = commands.add_parser("perturb")
    command.add_argument("manifest")
    command.add_argument("split")
    command.add_argument("output_dir")
    command.add_argument("operations", nargs="+")

    command = commands.add_parser("evaluate")
    command.add_argument("root")
    command.add_argument("checkpoint")
    command.add_argument("--name", required=True)
    command.add_argument("--test", action="append", default=[], help="name=manifest:split")
    command.add_argument("--val", action="store_true", help="Also score the run's own validation split")
    command.add_argument("--threads", type=int, default=12)

    command = commands.add_parser("summarise")
    command.add_argument("root")
    command.add_argument("--samples", type=int, default=1000)

    command = commands.add_parser("spectra")
    command.add_argument("output_dir")
    command.add_argument("datasets", nargs="+", help="name=manifest:split")
    command.add_argument("--per-class", type=int, default=500)

    command = commands.add_parser("occlusion")
    command.add_argument("checkpoint")
    command.add_argument("manifest")
    command.add_argument("split")
    command.add_argument("output_dir")
    command.add_argument("--per-class", type=int, default=100)
    command.add_argument("--threads", type=int, default=12)

    command = commands.add_parser("subset")
    command.add_argument("manifest")
    command.add_argument("per_class", type=int)
    command.add_argument("output")

    command = commands.add_parser("merge")
    command.add_argument("output")
    command.add_argument("sources", nargs="+", help="manifest:train_per_class:val_per_class")

    args = parser.parse_args()
    if args.command == "perturb":
        from src.data.perturb import perturb_manifest

        print(perturb_manifest(args.manifest, args.split, args.output_dir, args.operations), "images")
    elif args.command == "evaluate":
        from src.runtime import load_checkpoint
        from src.runtime import seed_everything
        from src.suite import evaluate_checkpoint

        seed_everything(0, True, args.threads)
        tests = dict(parse_set(t) for t in args.test)
        if args.val:
            state = load_checkpoint(args.checkpoint)
            tests = {"val": (None, state["config"]["data"]["val_split"]), **tests}
        results = evaluate_checkpoint(args.checkpoint, tests, args.root, args.name)
        for name, metrics in results.items():
            print(f"{args.name:28} {name:22} auroc={metrics['auroc']:.4f} tpr@5%fpr={metrics.get('tpr_at_fpr_5') or 0:.4f}")
    elif args.command == "summarise":
        from src.suite import summarise_root

        print(len(summarise_root(args.root, samples=args.samples)), "summaries")
    elif args.command == "spectra":
        from src.diagnostics import spectra

        print(json.dumps(spectra(dict(parse_set(d) for d in args.datasets), args.output_dir, args.per_class), indent=2))
    elif args.command == "occlusion":
        from src.diagnostics import occlusion

        torch.set_num_threads(args.threads)
        stats = occlusion(args.checkpoint, args.manifest, args.split, args.output_dir, args.per_class)
        print({k: round(v["border_over_interior_abs"], 3) for k, v in stats.items()})
    elif args.command == "subset":
        print(subset(args.manifest, args.per_class, args.output))
    elif args.command == "merge":
        print(json.dumps(merge(args.output, args.sources), indent=2))


if __name__ == "__main__":
    main()
