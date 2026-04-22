"""
MeCab ラッパー（fugashi + ipadic）

fugashi はシステムの libmecab に依存しないため、
Streamlit Community Cloud などの環境でも追加設定なしで動作する。
"""

from __future__ import annotations
from typing import List, Tuple

import fugashi
import ipadic

_wakati_tagger: fugashi.Tagger | None = None
_full_tagger: fugashi.Tagger | None = None


def _get_wakati() -> fugashi.Tagger:
    global _wakati_tagger
    if _wakati_tagger is None:
        _wakati_tagger = fugashi.Tagger(f"-Owakati {ipadic.MECAB_ARGS}")
    return _wakati_tagger


def _get_full() -> fugashi.Tagger:
    global _full_tagger
    if _full_tagger is None:
        _full_tagger = fugashi.Tagger(ipadic.MECAB_ARGS)
    return _full_tagger


def wakati(text: str) -> str:
    """空白区切りの分かち書きを返す"""
    return _get_wakati()(text).strip()


def lemmatize(text: str) -> str:
    """
    原形（lemma）の空白区切りを返す。
    lemmaが*の場合は表層形を使用。
    TF-IDF や語彙分析に利用。
    """
    tagger = _get_full()
    tokens: List[str] = []
    for word in tagger(text):
        parts = word.feature.split(",")
        if len(parts) > 6 and parts[6] != "*":
            tokens.append(parts[6])
        elif word.surface:
            tokens.append(word.surface)
    return " ".join(tokens)


def parse_pos(text: str) -> List[Tuple[str, str]]:
    """
    (surface, pos) のリストを返す。
    stylometry の品詞比率計算に利用。
    """
    tagger = _get_full()
    result: List[Tuple[str, str]] = []
    for word in tagger(text):
        if not word.surface:
            continue
        parts = word.feature.split(",")
        pos = parts[0] if parts else ""
        result.append((word.surface, pos))
    return result
