"""Fixed ontology used for triplet extraction — shared by the offline graph-builder
(over chunks) and the online query pipeline (over the user's question). Ported from
research/GetTripletsFromChunks.ipynb; keep this file as the single source of truth for
the entity-type list rather than duplicating it in prompts.
"""

ONTOLOGY_VERSION = "v1-hse-miem-2026"

BASE_ENTITY_TYPES: tuple[str, ...] = (
    "Студент",
    "Преподаватель",
    "Учебный офис",
    "Дисциплина",
    "Комиссия",
    "Скидка",
    "Экзамен",
    "Приказ",
    "Пересдача",
    "Заявление",
)
