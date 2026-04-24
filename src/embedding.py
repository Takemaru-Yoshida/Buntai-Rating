"""埋め込みモデルとコサイン類似度評価モジュール

使用モデル: intfloat/multilingual-e5-base
  - XLM-RoBERTa ベース、SentencePiece トークナイザー
  - MeCab / fugashi に依存しないため Streamlit Cloud (Python 3.14) でも動作する
  - 日本語を含む多言語テキストに対して高品質な埋め込みを生成する

実装上の注意:
  transformers のトップレベルインポートは Streamlit 起動時のモジュールスキャンが
  torchvision を要求して segfault を起こすため遅延インポートする。
"""

from __future__ import annotations
from typing import Dict, List, Tuple
from statistics import mean, stdev
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
from scipy.stats import norm

MODEL_NAME = "intfloat/multilingual-e5-base"
_tokenizer = None
_model = None


def _get_model():
    global _tokenizer, _model
    if _model is None:
        from transformers import AutoTokenizer, AutoModel
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model = AutoModel.from_pretrained(MODEL_NAME)
        _model.eval()
    return _tokenizer, _model


def _mean_pooling(last_hidden_state, attention_mask):
    import torch
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    return torch.sum(last_hidden_state * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


def encode_texts(
    texts: List[str],
    batch_size: int = 32,
    show_progress: bool = False,
) -> np.ndarray:
    """テキストリストをエンコードして (N, dim) ndarray を返す"""
    import torch
    import torch.nn.functional as F

    tokenizer, model = _get_model()
    all_vecs: List[np.ndarray] = []
    indices = range(0, len(texts), batch_size)
    if show_progress:
        from tqdm import tqdm
        indices = tqdm(indices, desc="encoding")
    for i in indices:
        batch = texts[i: i + batch_size]
        encoded = tokenizer(
            batch, padding=True, truncation=True,
            max_length=512, return_tensors="pt",
        )
        with torch.no_grad():
            output = model(**encoded)
        vecs = _mean_pooling(output.last_hidden_state, encoded["attention_mask"])
        vecs = F.normalize(vecs, p=2, dim=1)
        all_vecs.append(vecs.cpu().numpy())
    return np.vstack(all_vecs)


def encode_single(text: str) -> np.ndarray:
    """1件テキストをエンコードして1次元 ndarray を返す"""
    return encode_texts([text], batch_size=1)[0]


def build_author_embedding_db(
    dic: Dict[int, Dict[int, str]],
    batch_size: int = 32,
    show_progress: bool = True,
) -> Dict[int, np.ndarray]:
    """
    全作家・作品の embedding を計算して返す。
    dic: {author_idx: {work_idx: raw_text}}
    Returns: {author_idx: ndarray of shape (n_works, dim)}
    """
    _get_model()
    result: Dict[int, np.ndarray] = {}
    for author_idx, works in dic.items():
        texts = list(works.values())
        result[author_idx] = encode_texts(texts, batch_size=batch_size, show_progress=show_progress)
    return result


def calculate_similarity(
    query_vec: np.ndarray,
    author_vecs: Dict[int, np.ndarray],
    n_authors: int = 15,
) -> Tuple[List[float], List[float]]:
    """
    クエリ embedding と各作家 embedding のコサイン類似度を計算。
    Returns: (sim_mean, sim_max) — 各長さ n_authors のリスト (正規化済スコア 0-1)
    """
    raw_mean: List[float] = []
    raw_max: List[float] = []
    all_sims: List[float] = []

    for idx in range(n_authors):
        vecs = author_vecs[idx]
        sims = [
            float(np.dot(query_vec, v) / (np.linalg.norm(query_vec) * np.linalg.norm(v) + 1e-9))
            for v in vecs
        ]
        raw_mean.append(mean(sims))
        raw_max.append(max(sims))
        all_sims.extend(sims)

    mu_mean, std_mean = mean(raw_mean), stdev(raw_mean)
    mu_all, std_all = mean(all_sims), stdev(all_sims)

    sim_mean = [norm.cdf(v, mu_mean, std_mean) for v in raw_mean]
    sim_max = [norm.cdf(v, mu_all, std_all) for v in raw_max]

    return sim_mean, sim_max


def score_to_point(score: float) -> int:
    """0-1 スコアを 0-100 点整数に変換"""
    return int(Decimal(str(score * 100)).quantize(Decimal("0"), ROUND_HALF_UP))
