"""文体計量分析モジュール（スタイロメトリー）"""

import re
import unicodedata
from typing import Dict
import MeCab


_tagger = None


def _get_tagger() -> MeCab.Tagger:
    global _tagger
    if _tagger is None:
        _tagger = MeCab.Tagger("")
        _tagger.parse("")
    return _tagger


def _is_kanji(ch: str) -> bool:
    return unicodedata.category(ch) == "Lo" and "\u4e00" <= ch <= "\u9fff"


def _is_hiragana(ch: str) -> bool:
    return "\u3041" <= ch <= "\u309f"


def _is_katakana(ch: str) -> bool:
    return "\u30a0" <= ch <= "\u30ff"


def extract_stylometric_features(text: str) -> Dict[str, float]:
    """
    テキストから文体計量特徴を抽出して辞書で返す。

    特徴一覧:
    - kanji_ratio          漢字率
    - hiragana_ratio       ひらがな率
    - katakana_ratio       カタカナ率
    - avg_sentence_len     平均文長（文字数）
    - avg_word_len         平均語長（文字数）
    - lexical_diversity    語彙豊富度（TTR: type-token ratio）
    - punctuation_density  句読点密度（句読点数 / 総文字数）
    - avg_clauses_per_sent 1文あたりの読点数（節の多さの代理指標）
    - noun_ratio           名詞率
    - verb_ratio           動詞率
    - adjective_ratio      形容詞率
    - adverb_ratio         副詞率
    - particle_ratio       助詞率
    - char_count           総文字数
    """
    chars = re.sub(r"\s", "", text)
    if not chars:
        return {}

    total_chars = len(chars)
    kanji_count = sum(1 for c in chars if _is_kanji(c))
    hiragana_count = sum(1 for c in chars if _is_hiragana(c))
    katakana_count = sum(1 for c in chars if _is_katakana(c))

    # 文分割（。！？で分割）
    sentences = [s.strip() for s in re.split(r"[。！？]", text) if s.strip()]
    avg_sentence_len = (
        sum(len(re.sub(r"\s", "", s)) for s in sentences) / len(sentences)
        if sentences else 0.0
    )

    # 句読点
    punct_count = len(re.findall(r"[、。！？・…]", text))
    touten_count = len(re.findall(r"、", text))

    avg_clauses_per_sent = (
        touten_count / len(sentences) if sentences else 0.0
    )

    # 形態素解析
    tagger = _get_tagger()
    node = tagger.parseToNode(text)
    words = []
    pos_counts: Dict[str, int] = {}
    while node:
        if node.surface:
            words.append(node.surface)
            pos = node.feature.split(",")[0]
            pos_counts[pos] = pos_counts.get(pos, 0) + 1
        node = node.next

    word_count = len(words)
    avg_word_len = (
        sum(len(w) for w in words) / word_count if word_count else 0.0
    )
    unique_words = len(set(words))
    lexical_diversity = unique_words / word_count if word_count else 0.0

    def pos_ratio(pos_name: str) -> float:
        return pos_counts.get(pos_name, 0) / word_count if word_count else 0.0

    return {
        "kanji_ratio": kanji_count / total_chars,
        "hiragana_ratio": hiragana_count / total_chars,
        "katakana_ratio": katakana_count / total_chars,
        "avg_sentence_len": avg_sentence_len,
        "avg_word_len": avg_word_len,
        "lexical_diversity": lexical_diversity,
        "punctuation_density": punct_count / total_chars,
        "avg_clauses_per_sent": avg_clauses_per_sent,
        "noun_ratio": pos_ratio("名詞"),
        "verb_ratio": pos_ratio("動詞"),
        "adjective_ratio": pos_ratio("形容詞"),
        "adverb_ratio": pos_ratio("副詞"),
        "particle_ratio": pos_ratio("助詞"),
        "char_count": float(total_chars),
    }


FEATURE_LABELS_JA = {
    "kanji_ratio": "漢字率",
    "hiragana_ratio": "ひらがな率",
    "katakana_ratio": "カタカナ率",
    "avg_sentence_len": "平均文長",
    "avg_word_len": "平均語長",
    "lexical_diversity": "語彙豊富度",
    "punctuation_density": "句読点密度",
    "avg_clauses_per_sent": "1文あたり読点数",
    "noun_ratio": "名詞率",
    "verb_ratio": "動詞率",
    "adjective_ratio": "形容詞率",
    "adverb_ratio": "副詞率",
    "particle_ratio": "助詞率",
}
