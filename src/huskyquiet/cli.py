"""Command-line entry point: ``python -m huskyquiet <stage>`` or ``huskyquiet <stage>``."""

from __future__ import annotations

import argparse
import logging
import sys

from huskyquiet.config import get_paths
from huskyquiet.logging_utils import setup_logging

log = logging.getLogger("huskyquiet")


def _download(args: argparse.Namespace) -> None:
    from huskyquiet.data.ingest import download_permits

    download_permits(page_size=args.page_size)


def _clean(_: argparse.Namespace) -> None:
    from huskyquiet.data.clean import run_clean

    run_clean()


def _audit(_: argparse.Namespace) -> None:
    from huskyquiet.data.audit import run_audit

    run_audit()


def _snapshots(_: argparse.Namespace) -> None:
    from huskyquiet.features.snapshots import run_build_snapshots

    run_build_snapshots()


def _train(args: argparse.Namespace) -> None:
    from huskyquiet.models.train import run_training

    run_training(run_backtest=not args.skip_backtest)


def _all(args: argparse.Namespace) -> None:
    if not args.skip_download or not get_paths().raw_permits.exists():
        _download(args)
    _clean(args)
    _audit(args)
    _snapshots(args)
    _train(args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="huskyquiet", description=__doc__)
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-file", default=None, help="Also append logs to this file")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("download", help="Download raw permits from data.seattle.gov")
    p.add_argument("--page-size", type=int, default=50_000)
    p.set_defaults(func=_download)

    sub.add_parser("clean", help="Validate, type, and deduplicate raw data").set_defaults(func=_clean)
    sub.add_parser("audit", help="Compute data-quality statistics").set_defaults(func=_audit)
    sub.add_parser("snapshots", help="Build forecast-date snapshots").set_defaults(func=_snapshots)

    p = sub.add_parser("train", help="Train, evaluate, calibrate, and persist models")
    p.add_argument("--skip-backtest", action="store_true")
    p.set_defaults(func=_train)

    p = sub.add_parser("all", help="Run every stage end to end")
    p.add_argument("--page-size", type=int, default=50_000)
    p.add_argument("--skip-download", action="store_true", help="Reuse existing raw data")
    p.add_argument("--skip-backtest", action="store_true")
    p.set_defaults(func=_all)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    setup_logging(args.log_level.upper(), args.log_file)
    try:
        args.func(args)
    except Exception:
        log.exception("Stage '%s' failed", args.command)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
