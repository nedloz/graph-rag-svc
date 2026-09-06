"""Операции ручного курирования графа.

Живут здесь, а не в db-svc, по двум причинам: правила резолюции и upsert'а уже реализованы
в `app/graph/`, и дублировать их на голом SQL в админке значило бы гарантированно разъехаться
с билдером; а сам граф принадлежит соседней команде, которая в db-svc не ориентируется.
db-svc выступает только клиентом HTTP-API поверх этого модуля. См. README, §13.3.

Ключевое требование ко всем операциям: правка должна пережить пересборку графа. Билдер
работает через upsert по естественным ключам, поэтому:
  * склейка переносит связи и провенанс на целевую сущность, а имя удаляемой записывает её
    алиасом — тогда следующее извлечение найдёт строку по алиасу и вернёт целевую сущность,
    то есть операция самозалечивается;
  * смена типа помечает сущность как курированную (`attributes_json.curated`);
  * удаление сопровождается записью в стоп-лист, иначе сущность будет создана заново.
"""

import uuid
from typing import Any

from sqlalchemy import delete, func, or_, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    CurationBlocklist,
    Entity,
    EntityAlias,
    EntitySourceChunk,
    ExtractionRun,
    LibraryChunk,
    LibraryDocument,
    Relation,
)
from app.graph.resolution import normalize_alias


class CurationError(Exception):
    """Некорректная операция курирования (несуществующая сущность, склейка сама с собой…)."""


# --------------------------------------------------------------------------- чтение


async def stats(session: AsyncSession) -> dict[str, Any]:
    by_type = (
        await session.execute(
            select(Entity.entity_type, func.count()).group_by(Entity.entity_type).order_by(func.count().desc())
        )
    ).all()
    relations_total = (await session.execute(select(func.count()).select_from(Relation))).scalar_one()
    aliases_total = (await session.execute(select(func.count()).select_from(EntityAlias))).scalar_one()
    blocked_total = (await session.execute(select(func.count()).select_from(CurationBlocklist))).scalar_one()

    runs = (
        await session.execute(
            select(ExtractionRun).order_by(ExtractionRun.started_at.desc()).limit(5)
        )
    ).scalars().all()

    return {
        "entities_total": sum(count for _, count in by_type),
        "entities_by_type": [{"entity_type": t, "count": c} for t, c in by_type],
        "relations_total": relations_total,
        "aliases_total": aliases_total,
        "blocklist_total": blocked_total,
        "recent_runs": [
            {
                "id": str(run.id),
                "run_type": run.run_type,
                "status": run.status,
                "ontology_version": run.ontology_version,
                "stats": run.stats_json,
                "started_at": run.started_at.isoformat() if run.started_at else None,
                "finished_at": run.finished_at.isoformat() if run.finished_at else None,
            }
            for run in runs
        ],
    }


async def search_entities(
    session: AsyncSession,
    *,
    query: str | None = None,
    entity_type: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict[str, Any]]:
    stmt = select(Entity)
    if entity_type:
        stmt = stmt.where(Entity.entity_type == entity_type)
    if query:
        pattern = f"%{query.strip()}%"
        alias_match = select(EntityAlias.entity_id).where(EntityAlias.alias_normalized.ilike(f"%{normalize_alias(query)}%"))
        stmt = stmt.where(or_(Entity.canonical_name.ilike(pattern), Entity.id.in_(alias_match)))

    stmt = stmt.order_by(Entity.canonical_name).limit(limit).offset(offset)
    entities = (await session.execute(stmt)).scalars().all()
    return [
        {
            "id": str(e.id),
            "entity_type": e.entity_type,
            "canonical_name": e.canonical_name,
            "curated": bool((e.attributes_json or {}).get("curated")),
        }
        for e in entities
    ]


async def entity_card(session: AsyncSession, entity_id: uuid.UUID) -> dict[str, Any]:
    """Карточка сущности: алиасы, связи в обе стороны и чанки-источники с текстом.

    Тексты чанков и названия документов нужны, чтобы куратор мог судить о корректности
    извлечения, а не только видеть голые идентификаторы.
    """
    entity = await session.get(Entity, entity_id)
    if entity is None:
        raise CurationError(f"сущность {entity_id} не найдена")

    aliases = (
        await session.execute(select(EntityAlias).where(EntityAlias.entity_id == entity_id).order_by(EntityAlias.alias))
    ).scalars().all()

    subject_alias = Entity.__table__.alias("subj")
    object_alias = Entity.__table__.alias("obj")
    relation_rows = (
        await session.execute(
            select(
                Relation.id,
                Relation.relation,
                Relation.confidence,
                Relation.subject_entity_id,
                Relation.object_entity_id,
                subject_alias.c.canonical_name.label("subject_name"),
                object_alias.c.canonical_name.label("object_name"),
            )
            .join(subject_alias, subject_alias.c.id == Relation.subject_entity_id)
            .join(object_alias, object_alias.c.id == Relation.object_entity_id)
            .where(or_(Relation.subject_entity_id == entity_id, Relation.object_entity_id == entity_id))
            .order_by(Relation.relation)
        )
    ).mappings().all()

    chunk_rows = (
        await session.execute(
            select(LibraryChunk.id, LibraryChunk.text, LibraryDocument.id.label("doc_id"), LibraryDocument.title)
            .join(EntitySourceChunk, EntitySourceChunk.chunk_id == LibraryChunk.id)
            .join(LibraryDocument, LibraryDocument.id == LibraryChunk.document_id)
            .where(EntitySourceChunk.entity_id == entity_id)
            .distinct()
            .limit(50)
        )
    ).mappings().all()

    return {
        "id": str(entity.id),
        "entity_type": entity.entity_type,
        "canonical_name": entity.canonical_name,
        "attributes_json": entity.attributes_json or {},
        "curated": bool((entity.attributes_json or {}).get("curated")),
        "embedding_model": entity.embedding_model,
        "aliases": [
            {"id": str(a.id), "alias": a.alias, "alias_normalized": a.alias_normalized, "source": a.source}
            for a in aliases
        ],
        "relations": [
            {
                "id": str(r["id"]),
                "relation": r["relation"],
                "confidence": r["confidence"],
                "direction": "outgoing" if r["subject_entity_id"] == entity_id else "incoming",
                "subject": {"id": str(r["subject_entity_id"]), "name": r["subject_name"]},
                "object": {"id": str(r["object_entity_id"]), "name": r["object_name"]},
            }
            for r in relation_rows
        ],
        "source_chunks": [
            {
                "chunk_id": str(c["id"]),
                "doc_id": str(c["doc_id"]),
                "doc_title": c["title"],
                "text": c["text"],
            }
            for c in chunk_rows
        ],
    }


async def orphans(session: AsyncSession, limit: int = 200) -> dict[str, Any]:
    """Кандидаты на вычистку — та самая проблема «граф только растёт».

    `without_relations` — сущности, ни в одной связи не участвующие.
    `without_sources` — сущности, потерявшие провенанс: `entity_source_chunks.chunk_id`
    объявлен с ON DELETE CASCADE, поэтому переиндексация документа удаляет старые чанки и
    вместе с ними строки провенанса. Ноль источников означает, что сущность извлечена из
    текста, которого в базе больше нет.
    """
    related = select(Relation.subject_entity_id).union(select(Relation.object_entity_id))
    without_relations = (
        await session.execute(
            select(Entity.id, Entity.canonical_name, Entity.entity_type)
            .where(Entity.id.notin_(related))
            .limit(limit)
        )
    ).mappings().all()

    with_sources = select(EntitySourceChunk.entity_id).where(EntitySourceChunk.entity_id.isnot(None))
    without_sources = (
        await session.execute(
            select(Entity.id, Entity.canonical_name, Entity.entity_type)
            .where(Entity.id.notin_(with_sources))
            .limit(limit)
        )
    ).mappings().all()

    def _fmt(rows):
        return [{"id": str(r["id"]), "canonical_name": r["canonical_name"], "entity_type": r["entity_type"]} for r in rows]

    return {"without_relations": _fmt(without_relations), "without_sources": _fmt(without_sources)}


# --------------------------------------------------------------------------- запись


async def update_entity(
    session: AsyncSession,
    entity_id: uuid.UUID,
    *,
    canonical_name: str | None = None,
    entity_type: str | None = None,
    attributes_json: dict | None = None,
) -> dict[str, Any]:
    entity = await session.get(Entity, entity_id)
    if entity is None:
        raise CurationError(f"сущность {entity_id} не найдена")

    if canonical_name:
        entity.canonical_name = canonical_name.strip()
    if entity_type:
        entity.entity_type = entity_type.strip()
    attributes = dict(entity.attributes_json or {})
    if attributes_json:
        attributes.update(attributes_json)
    # Пометка ручного вмешательства: билдер сейчас поля существующих сущностей не
    # перезаписывает, но флаг нужен, чтобы это осталось верным и дальше, и чтобы куратор
    # видел, что запись правили руками.
    attributes["curated"] = True
    entity.attributes_json = attributes

    await session.flush()
    return {"id": str(entity.id), "entity_type": entity.entity_type, "canonical_name": entity.canonical_name}


async def add_alias(session: AsyncSession, entity_id: uuid.UUID, alias: str) -> dict[str, Any]:
    entity = await session.get(Entity, entity_id)
    if entity is None:
        raise CurationError(f"сущность {entity_id} не найдена")

    alias_normalized = normalize_alias(alias)
    if not alias_normalized:
        raise CurationError("пустой алиас")

    existing = (
        await session.execute(
            select(EntityAlias).where(EntityAlias.alias_normalized == alias_normalized).limit(1)
        )
    ).scalars().first()
    if existing is not None:
        if existing.entity_id == entity_id:
            return {"id": str(existing.id), "alias": existing.alias}
        raise CurationError(f"алиас '{alias}' уже принадлежит другой сущности {existing.entity_id}")

    row = EntityAlias(entity_id=entity_id, alias=alias.strip(), alias_normalized=alias_normalized, source="curation")
    session.add(row)
    await session.flush()
    return {"id": str(row.id), "alias": row.alias}


async def delete_alias(session: AsyncSession, entity_id: uuid.UUID, alias_id: uuid.UUID) -> None:
    row = await session.get(EntityAlias, alias_id)
    if row is None or row.entity_id != entity_id:
        raise CurationError("алиас не найден у этой сущности")
    await session.delete(row)
    await session.flush()


async def _repoint_source_chunks(session: AsyncSession, *, column, old_id: uuid.UUID, new_id: uuid.UUID) -> None:
    """Переносит строки провенанса, не создавая дубликатов (уникального индекса там нет)."""
    already = select(EntitySourceChunk.chunk_id).where(column == new_id)
    await session.execute(delete(EntitySourceChunk).where(column == old_id, EntitySourceChunk.chunk_id.in_(already)))
    await session.execute(update(EntitySourceChunk).where(column == old_id).values({column.key: new_id}))


async def merge_entities(session: AsyncSession, source_id: uuid.UUID, target_id: uuid.UUID) -> dict[str, Any]:
    """Склеивает `source` в `target`, оставляя имя источника алиасом целевой сущности.

    Именно алиас делает операцию устойчивой к пересборке: следующее извлечение того же имени
    разрешится по алиасу в целевую сущность, а не создаст дубликат заново.
    """
    if source_id == target_id:
        raise CurationError("нельзя склеить сущность саму с собой")

    source = await session.get(Entity, source_id)
    target = await session.get(Entity, target_id)
    if source is None or target is None:
        raise CurationError("одна из сущностей не найдена")

    source_name = source.canonical_name

    # 1. Связи. Дубликаты по (subject, object, relation) удаляем, а не обновляем — иначе
    #    нарушится уникальный индекс.
    relations = (
        await session.execute(
            select(Relation).where(or_(Relation.subject_entity_id == source_id, Relation.object_entity_id == source_id))
        )
    ).scalars().all()

    for rel in relations:
        new_subject = target_id if rel.subject_entity_id == source_id else rel.subject_entity_id
        new_object = target_id if rel.object_entity_id == source_id else rel.object_entity_id

        if new_subject == new_object:
            # После склейки связь превратилась бы в петлю — смысла в ней нет.
            await session.execute(delete(EntitySourceChunk).where(EntitySourceChunk.relation_id == rel.id))
            await session.delete(rel)
            continue

        duplicate_id = (
            await session.execute(
                select(Relation.id).where(
                    Relation.subject_entity_id == new_subject,
                    Relation.object_entity_id == new_object,
                    Relation.relation == rel.relation,
                    Relation.id != rel.id,
                )
            )
        ).scalars().first()

        if duplicate_id is not None:
            await _repoint_source_chunks(
                session, column=EntitySourceChunk.relation_id, old_id=rel.id, new_id=duplicate_id
            )
            await session.delete(rel)
        else:
            rel.subject_entity_id = new_subject
            rel.object_entity_id = new_object
        await session.flush()

    # 2. Провенанс сущности.
    await _repoint_source_chunks(session, column=EntitySourceChunk.entity_id, old_id=source_id, new_id=target_id)

    # 3. Алиасы: переносим те, которых у цели ещё нет.
    target_aliases = set(
        (await session.execute(select(EntityAlias.alias_normalized).where(EntityAlias.entity_id == target_id)))
        .scalars()
        .all()
    )
    source_aliases = (
        await session.execute(select(EntityAlias).where(EntityAlias.entity_id == source_id))
    ).scalars().all()
    for alias_row in source_aliases:
        if alias_row.alias_normalized in target_aliases:
            await session.delete(alias_row)
        else:
            alias_row.entity_id = target_id
            target_aliases.add(alias_row.alias_normalized)
    await session.flush()

    # 4. Имя удаляемой сущности — алиасом целевой. Это и есть самозалечивание.
    source_name_normalized = normalize_alias(source_name)
    if source_name_normalized and source_name_normalized not in target_aliases:
        session.add(
            EntityAlias(
                entity_id=target_id,
                alias=source_name,
                alias_normalized=source_name_normalized,
                source="curation_merge",
            )
        )

    await session.delete(source)
    await session.flush()

    return {
        "merged_into": str(target_id),
        "target_name": target.canonical_name,
        "alias_added": source_name,
        "relations_processed": len(relations),
    }


async def delete_entity(
    session: AsyncSession, entity_id: uuid.UUID, *, blocklist: bool = False, reason: str | None = None
) -> dict[str, Any]:
    """Удаляет сущность; с `blocklist=True` заносит её имена в стоп-лист.

    Без стоп-листа удаление бессмысленно как операция курирования: следующая пересборка
    извлечёт то же имя и создаст сущность заново.
    """
    entity = await session.get(Entity, entity_id)
    if entity is None:
        raise CurationError(f"сущность {entity_id} не найдена")

    blocked: list[str] = []
    if blocklist:
        names = {normalize_alias(entity.canonical_name)}
        aliases = (
            await session.execute(select(EntityAlias.alias_normalized).where(EntityAlias.entity_id == entity_id))
        ).scalars().all()
        names.update(a for a in aliases if a)

        existing = set(
            (
                await session.execute(
                    select(CurationBlocklist.name_normalized).where(CurationBlocklist.name_normalized.in_(names))
                )
            )
            .scalars()
            .all()
        )
        for name in sorted(names - existing):
            if not name:
                continue
            session.add(CurationBlocklist(name_normalized=name, reason=reason))
            blocked.append(name)

    await session.delete(entity)
    await session.flush()
    return {"deleted": str(entity_id), "blocklisted": blocked}


async def update_relation(
    session: AsyncSession, relation_id: uuid.UUID, *, relation: str | None = None, confidence: float | None = None
) -> dict[str, Any]:
    row = await session.get(Relation, relation_id)
    if row is None:
        raise CurationError(f"связь {relation_id} не найдена")
    if relation:
        row.relation = relation.strip()
    if confidence is not None:
        row.confidence = confidence
    await session.flush()
    return {"id": str(row.id), "relation": row.relation, "confidence": row.confidence}


async def delete_relation(session: AsyncSession, relation_id: uuid.UUID) -> None:
    row = await session.get(Relation, relation_id)
    if row is None:
        raise CurationError(f"связь {relation_id} не найдена")
    await session.execute(delete(EntitySourceChunk).where(EntitySourceChunk.relation_id == relation_id))
    await session.delete(row)
    await session.flush()


async def list_blocklist(session: AsyncSession, limit: int = 200) -> list[dict[str, Any]]:
    rows = (
        await session.execute(select(CurationBlocklist).order_by(CurationBlocklist.created_at.desc()).limit(limit))
    ).scalars().all()
    return [
        {"id": str(r.id), "name_normalized": r.name_normalized, "reason": r.reason}
        for r in rows
    ]


async def remove_from_blocklist(session: AsyncSession, entry_id: uuid.UUID) -> None:
    row = await session.get(CurationBlocklist, entry_id)
    if row is None:
        raise CurationError("запись стоп-листа не найдена")
    await session.delete(row)
    await session.flush()
