"""Command line entry point."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
from .config import ConfigError, load_config
from .dataset import DatasetError, check_data
from .models import UnsupportedModelError, require_supported

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m htrtrain")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("doctor", "train", "eval", "ocr"):
        sp = sub.add_parser(command)
        sp.add_argument("--config", required=True)
        sp.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
        if command == "train":
            sp.add_argument("--dry-run", action="store_true", help="Check two dataset samples on CPU; never load a model")
            sp.add_argument("--gpu-smoke", action="store_true", help="Isolated one-step forward/backward/optimizer/save/reload check")
            sp.add_argument("--resume", type=Path)
            sp.add_argument("--warm-start-from-adapter", type=Path)
        elif command == "eval":
            sp.add_argument("--split", choices=["test", "validation"], default="test")
            sp.add_argument("--limit", type=int)
            sp.add_argument("--adapter", type=Path)
            sp.add_argument("--compare-base", action="store_true")
        elif command == "ocr":
            sp.add_argument("--image", type=Path, required=True)
            sp.add_argument("--adapter", type=Path)
    data = sub.add_parser("data-check")
    data_source = data.add_mutually_exclusive_group(required=True)
    data_source.add_argument("--data-dir", type=Path)
    data_source.add_argument("--config")
    data.add_argument("--set", dest="overrides", action="append", default=[], metavar="KEY=VALUE")
    data.add_argument("--limit", type=int)
    prepare = sub.add_parser("data-prepare", help="Explicitly audit and create a portable clean-data directory")
    prepare.add_argument("--source", type=Path, required=True)
    prepare.add_argument("--dest", type=Path, required=True)
    return p

def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    try:
        if hasattr(args, "limit") and args.limit is not None and args.limit <= 0:
            raise ValueError("--limit must be positive")
        if args.command == "data-check":
            root = args.data_dir.resolve() if args.data_dir else load_config(args.config, args.overrides).data_dir
            result = check_data(root, limit=args.limit)
            ok = result["ok"]
        elif args.command == "data-prepare":
            from .prepare import prepare_data
            result = prepare_data(args.source, args.dest)
            ok = True
        else:
            config = load_config(args.config, args.overrides)
            require_supported(config)
            if args.command == "doctor":
                from .hardware import doctor
                result = doctor(config)
                ok = result["ok"]
            elif args.command == "train":
                from .train import train
                result = train(config, args.resume, args.warm_start_from_adapter, args.dry_run, args.gpu_smoke)
                ok = True
            elif args.command == "eval":
                from .evaluate import benchmark
                result = benchmark(config, args.split, args.limit, args.adapter or config.optional_path("adapter_dir"), args.compare_base)
                ok = True
            else:
                from .checkpoints import validate_adapter
                from .models import load_adapter, load_base
                from .inference import predict
                if not args.image.is_file():
                    raise ValueError(f"Image not found: {args.image}")
                adapter = args.adapter or config.optional_path("adapter_dir")
                if adapter:
                    validate_adapter(adapter, config)
                from .hardware import doctor
                report = doctor(config)
                if not report["ok"]:
                    raise RuntimeError("GPU preflight failed: " + "; ".join(report["errors"]))
                model, processor = load_adapter(config, adapter) if adapter else load_base(config)
                result = {"text": predict(model, processor, args.image, config)}
                ok = True
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if ok else 1
    except (ConfigError, DatasetError, UnsupportedModelError, ValueError, RuntimeError, OSError, ImportError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
