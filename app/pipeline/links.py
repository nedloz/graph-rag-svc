"""Релевантные ссылки для ответа (ячейка 15, `find_best_links`)."""

from sqlalchemy import select

from app.db.models import LibraryLink

LINKS_TOP_K = 3
LINKS_THRESHOLD = 0.75


async def find_links(
    session,
    query_vector: list[float],
    model: str,
    top_k: int = LINKS_TOP_K,
    threshold: float = LINKS_THRESHOLD,
) -> list[tuple[str, str]]:
    """До `top_k` ссылок с косинусной близостью описания строго больше `threshold`: [(описание, url)].

    URL отдаётся как записан в базе: ссылки с `http://` не теряются (ноутбук пропускал всё,
    кроме `https://`, ещё при чтении файла).
    """
    distance = LibraryLink.embedding.cosine_distance(query_vector)
    stmt = (
        select(LibraryLink.description, LibraryLink.url, (1 - distance).label("cosine"))
        .where(LibraryLink.is_active.is_(True))
        .where(LibraryLink.embedding.is_not(None))
        # Векторы другой модели несравнимы с вектором вопроса (см. index.dense_search).
        .where(LibraryLink.embedding_model == model)
        .order_by(distance)
        .limit(top_k)
    )
    rows = (await session.execute(stmt)).all()
    return [(row.description, row.url) for row in rows if float(row.cosine) > threshold]
