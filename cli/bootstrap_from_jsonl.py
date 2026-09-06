"""One-time importer for research/chunks_part_1.jsonl (triplets already extracted by
the original notebook) — resolution + storage only, extraction is skipped since the
triplets already exist.

Use this ONLY to smoke-test app/graph/{resolution,store}.py during development, not as
the production graph: the JSONL's `source_chunks` are small integer indices from the
notebook's own one-off chunking pass, not the UUIDs in the live `library.chunks` table,
so provenance cannot be linked and is intentionally NOT imported (see graph-rag-svc
README, "Оффлайн-пайплайн построения графа"). Run a real
`python -m cli.build_graph --full-rebuild` against current data for the actual graph.

Usage:
    python -m cli.bootstrap_from_jsonl research/chunks_part_1.jsonl [--limit N]
"""

import argparse
import asyncio
import json
import logging
import sys

from app.core.logging import configure_logging
from app.db.models import ExtractionRun
from app.db.session import SessionLocal
from app.graph.resolution import resolve_entity
from app.graph.store import create_extraction_run, finish_extraction_run, upsert_relation
from app.ontology.schema import ONTOLOGY_VERSION

logger = logging.getLogger("graph-rag-svc.cli.bootstrap_from_jsonl")

COMMIT_EVERY = 500


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl_path")
    parser.add_argument("--limit", type=int, default=None, help="Only import the first N lines (recommended for a smoke test)")
    return parser.parse_args(argv)


def _read_triplets(path: str, limit: int | None):
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                logger.warning("Skipping malformed JSONL line %d", i)
                continue
            subject, relation, obj = record.get("subject"), record.get("relation"), record.get("object")
            if subject and relation and obj:
                yield subject, relation, obj


async def _run(jsonl_path: str, limit: int | None) -> dict:
    stats = {"records_processed": 0, "relations_upserted": 0, "errors": 0}

    async with SessionLocal() as session:
        run = await create_extraction_run(
            session,
            run_type="bootstrap_jsonl",
            extraction_model=None,
            ontology_version=ONTOLOGY_VERSION,
        )
        await session.commit()
        run_id = run.id

    session = SessionLocal()
    try:
        for subject, relation, obj in _read_triplets(jsonl_path, limit):
            try:
                subject_entity = await resolve_entity(session, subject)
                object_entity = await resolve_entity(session, obj)
                await session.flush()
                await upsert_relation(
                    session,
                    subject_entity_id=subject_entity.id,
                    object_entity_id=object_entity.id,
                    relation=relation,
                    confidence=None,
                    extraction_run_id=run_id,
                )
                stats["relations_upserted"] += 1
            except Exception:
                logger.exception("Failed on triplet (%r, %r, %r)", subject, relation, obj)
                stats["errors"] += 1

            stats["records_processed"] += 1
            if stats["records_processed"] % COMMIT_EVERY == 0:
                await session.commit()
                logger.info("Progress: %s", stats)
        await session.commit()
    finally:
        await session.close()

    async with SessionLocal() as session:
        run = await session.get(ExtractionRun, run_id)
        await finish_extraction_run(session, run, status="completed", stats=stats)
        await session.commit()

    return stats


def main(argv: list[str] | None = None) -> int:
    configure_logging()
    args = parse_args(argv if argv is not None else sys.argv[1:])
    stats = asyncio.run(_run(args.jsonl_path, args.limit))
    logger.info("Done: %s", stats)
    return 1 if stats["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
