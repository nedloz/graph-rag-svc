"""CLI entrypoint for the offline graph-builder.

Usage (inside the container):
    python -m cli.build_graph --full-rebuild
    python -m cli.build_graph --document-id <uuid> [--document-id <uuid> ...]

Via docker compose from the host:
    docker compose run --rm graph-rag-svc python -m cli.build_graph --full-rebuild

Not a standing container — run on demand (see graph-rag-svc README, "Оффлайн-пайплайн
построения графа": graph construction is deliberately decoupled from the live ingest
pipeline for now).
"""

import argparse
import asyncio
import logging
import sys
import uuid

from app.core.logging import configure_logging
from app.db.session import SessionLocal
from app.graph.builder import build_graph

logger = logging.getLogger("graph-rag-svc.cli.build_graph")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-rebuild", action="store_true", help="Process every ingest_status='indexed' document")
    parser.add_argument("--document-id", action="append", dest="document_ids", metavar="UUID", help="Limit to specific document(s); repeatable")
    args = parser.parse_args(argv)

    if not args.full_rebuild and not args.document_ids:
        parser.error("specify --full-rebuild or at least one --document-id")
    if args.full_rebuild and args.document_ids:
        parser.error("--full-rebuild and --document-id are mutually exclusive")
    return args


async def _run(args: argparse.Namespace) -> dict:
    document_ids = [uuid.UUID(d) for d in args.document_ids] if args.document_ids else None
    run_type = "full_rebuild" if args.full_rebuild else "incremental"
    return await build_graph(SessionLocal, document_ids=document_ids, run_type=run_type)


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv if argv is not None else sys.argv[1:])
    stats = asyncio.run(_run(args))
    logger.info("Done: %s", stats)
    return 1 if stats.get("errors") else 0


if __name__ == "__main__":
    raise SystemExit(main())
