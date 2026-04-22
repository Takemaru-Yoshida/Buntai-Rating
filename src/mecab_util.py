"""
MeCab ラッパー（fugashi + ipadic）

fugashi はシステムの libmecab に依存しないため、
Streamlit Community Cloud などの環境でも追加設定なしで動作する。

Note: fugashi.Tagger は ipadic では動作しないため GenericTagger を使用する。
      GenericTagger では word.feature がタプルで返る。
"""

from __future__ import annotations
from typing import List, Tuple

import fugashi
import ipadic

_wakati_tagger: fugashi.GenericTagger | None = None
_full_tagger: fugashi.GenericTagger | None = None


def _get_wakati() -> fugashi.GenericTagger:
    global _wakati_tagger
    if _wakati_tagger is None:
        _wakati_tagger = fugashi.GenericTagger(f"-Owakati {ipadic.MECAB_ARGS}")
    return _wakati_tagger


def _get_full() -> fugashi.GenericTagger:
    global _full_tagger
    if _full_tagger is None:
        _full_tagger = fugashi.GenericTagger(ipadic.MECAB_ARGS)
    return _full_tagger


def wakati(text: str) -> str:
    """空白区切りの分かち書きを返す"""
    if not text or not text.strip():
        return ""
    words = _get_wakati()(text)
    return " ".join(w.surface for w in words if w.surface)


def lemmatize(text: str) -> str:
    """
    原形（lemma）の空白区切りを返す。
    lemmaが*の場合は表層形を使用。
    TF-IDF や語彙分析に利用。
    """
    if not text or not text.strip():
        return ""
    tagger = _get_full()
    tokens: List[str] = []
    for word in tagger(text):
        if not word.surface:
            continue
        try:
            feat = word.feature  # タプル
            # ipadic: feature[6] = 原形
            if len(feat) > 6 and feat[6] not in ("", "*"):
                tokens.append(feat[6])
            else:
                tokens.append(word.surface)
        except Exception:
            tokens.append(word.surface)
    return " ".join(tokens) if tokens else ""


def parse_pos(text: str) -> List[Tuple[str, str]]:
    """
    (surface, pos) のリストを返す。
    stylometry の品詞比率計算に利用。
    """
    if not text or not text.strip():
        return []
    tagger = _get_full()
    result: List[Tuple[str, str]] = []
    for word in tagger(text):
        if not word.surface:
            continue
        try:
            feat = word.feature
            pos = feat[0] if feat else ""
        except Exception:
            pos = ""
        result.append((word.surface, pos))
    return result
