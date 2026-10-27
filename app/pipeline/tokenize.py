"""Токенизация для BM25 по чанкам (ячейка 13): слова длиннее двух символов, Snowball (ru).

SnowballStemmer не требует данных nltk (`nltk.download` не нужен).
"""

import re

from nltk.stem.snowball import SnowballStemmer

_WORD_RE = re.compile(r"[A-Za-zА-Яа-яЁё0-9]+")
_stemmer = SnowballStemmer("russian")


def tokenize(text: str) -> list[str]:
    result = []
    for word in _WORD_RE.findall(str(text).lower()):
        if len(word) <= 2:
            continue
        try:
            word = _stemmer.stem(word)
        except Exception:
            pass
        result.append(word)
    return result
