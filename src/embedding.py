"""埋め込みモデルとコサイン類似度評価モジュール

使用モデル: cl-nagoya/sup-simcse-ja-base
  - Supervised SimCSE（対照学習）で日本語NLIデータをファインチューニング
  - sonoisa/sentence-bert-base-ja-mean-tokens-v2 は新しい sentence-transformers と
    BertJapaneseTokenizer の do_lower_case 属性非互換があるため、
    こちらを採用（標準 SentenceTransformer API と完全互換）
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from statistics import mean, stdev
from scipy.stats import norm
from decimal import Decimal, ROUND_HALF_UP

import numpy as np

MODEL_NAME = "cl-nagoya/sup-simcse-ja-base"
_model = None


def get_model():
    """モデルをシングルトンで返す（初回のみダウンロード）。
    sentence_transformers は起動時ではなく初回呼び出し時にインポートする。
    トップレベルで import すると transformers 経由で torchvision が要求され
    segfault の原因になるため遅延インポートしている。
    """
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def encode_texts(texts: List[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    """テキストリストをエンコードしてndarrayを返す"""
    return get_model().encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=show_progress,
        convert_to_numpy=True,
    )


def encode_single(text: str) -> np.ndarray:
    """1件テキストをエンコードして1次元ndarrayを返す"""
    return get_model().encode(text, convert_to_numpy=True)


def build_author_embedding_db(
    dic: Dict[int, Dict[int, str]],
    batch_size: int = 32,
    show_progress: bool = True,
) -> Dict[int, np.ndarray]:
    """
    全作家・作品のembeddingを計算して返す。
    dic: {author_idx: {work_idx: raw_text}}
    Returns: {author_idx: ndarray of shape (n_works, dim)}
    """
    model = get_model()
    result: Dict[int, np.ndarray] = {}
    for author_idx, works in dic.items():
        texts = list(works.values())
        result[author_idx] = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
        )
    return result


def calculate_similarity(
    query_vec: np.ndarray,
    author_vecs: Dict[int, np.ndarray],
    n_authors: int = 15,
) -> Tuple[List[float], List[float]]:
    """
    クエリembeddingと各作家embeddingのコサイン類似度を計算。
    Returns: (sim_mean, sim_max) — 各長さ n_authors のリスト (正規化済スコア 0-1)
    """
    from sentence_transformers import util

    raw_mean: List[float] = []
    raw_max: List[float] = []
    all_sims: List[float] = []

    for idx in range(n_authors):
        vecs = author_vecs[idx]
        sims = [float(util.cos_sim(query_vec, v)) for v in vecs]
        raw_mean.append(mean(sims))
        raw_max.append(max(sims))
        all_sims.extend(sims)

    mu_mean, std_mean = mean(raw_mean), stdev(raw_mean)
    mu_all, std_all = mean(all_sims), stdev(all_sims)

    sim_mean = [norm.cdf(v, mu_mean, std_mean) for v in raw_mean]
    sim_max = [norm.cdf(v, mu_all, std_all) for v in raw_max]

    return sim_mean, sim_max


def score_to_point(score: float) -> int:
    """0-1スコアを0-100点整数に変換"""
    return int(Decimal(str(score * 100)).quantize(Decimal("0"), ROUND_HALF_UP))
