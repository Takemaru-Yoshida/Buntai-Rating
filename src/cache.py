"""
データキャッシュ管理モジュール

キャッシュ保存先（優先順位）:
  1. /tmp/buntai_cache/  ← Streamlit Community Cloud のエフェメラルストレージ
  2. ./data/             ← ローカル開発時

各処理ステップの結果を個別に pickle 保存するため、
途中で中断しても再起動時に続きから再開できる。
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, Optional

# Streamlit Cloud では /tmp が書き込み可能
# ローカルでは data/ を使う
_CACHE_CANDIDATES = ["/tmp/buntai_cache", os.path.join(os.path.dirname(__file__), "..", "data")]

CACHE_DIR: str = os.environ.get(
    "BUNTAI_CACHE_DIR",
    _CACHE_CANDIDATES[0] if os.access("/tmp", os.W_OK) else _CACHE_CANDIDATES[1],
)

REQUIRED_FILES = [
    "author_vecs.pkl",
    "df_vec_tfidf.pkl",
    "lgbm_tfidf.pkl",
    "stylometry.pkl",
]


def cache_path(filename: str) -> str:
    return os.path.join(CACHE_DIR, filename)


def is_step_done(filename: str) -> bool:
    return os.path.exists(cache_path(filename))


def is_setup_complete() -> bool:
    return all(is_step_done(f) for f in REQUIRED_FILES)


def save(obj: Any, filename: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path(filename), "wb") as f:
        pickle.dump(obj, f)


def load(filename: str) -> Optional[Any]:
    p = cache_path(filename)
    if not os.path.exists(p):
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


# ── セットアップ処理（各ステップ）────────────────────────────────────────────

def run_step_scrape(log=print) -> Dict:
    """青空文庫スクレイピング（最も時間がかかるステップ）"""
    from src.data import urls
    from src.text_processing import fetch_from_url

    corpus: Dict[int, Dict[int, str]] = load("corpus.pkl") or {}
    total_authors = len(urls)

    for idx, (author, work_urls) in enumerate(urls.items()):
        if idx in corpus:
            log(f"  [{idx + 1}/{total_authors}] {author}: スキップ（キャッシュ済み）")
            continue
        corpus[idx] = {}
        log(f"  [{idx + 1}/{total_authors}] {author}: {len(work_urls)} 作品を取得中...")
        for work_idx, url in enumerate(work_urls):
            try:
                corpus[idx][work_idx] = fetch_from_url(url)
                time.sleep(0.5)
            except Exception as e:
                log(f"    WARNING: {url} → {e}")
        save(corpus, "corpus.pkl")
        log(f"    → {len(corpus[idx])} 作品取得完了")

    return corpus


def run_step_parse(corpus: Dict, log=print) -> Dict:
    """形態素解析"""
    from src.text_processing import parsetext

    parsed: Dict[int, Dict[int, str]] = load("parsed.pkl") or {}

    # キャッシュに有効なテキストがなければ破棄して再解析
    def _has_valid_text(d: dict) -> bool:
        return any(v and v.strip() for v in d.values())

    if parsed and not any(_has_valid_text(v) for v in parsed.values()):
        log("  WARNING: parsed.pkl に有効テキストがないため削除して再解析します")
        p = cache_path("parsed.pkl")
        if os.path.exists(p):
            os.remove(p)
        parsed = {}

    for idx, works in corpus.items():
        if idx in parsed and _has_valid_text(parsed[idx]):
            continue
        parsed[idx] = {}
        for work_idx, text in works.items():
            try:
                result = parsetext(text)
                if result and result.strip():
                    parsed[idx][work_idx] = result
            except Exception as e:
                log(f"  WARNING: parse [{idx}][{work_idx}]: {e}")
        log(f"  作家 {idx} 解析完了 ({len(parsed[idx])} 作品)")
    save(parsed, "parsed.pkl")

    total = sum(1 for v in parsed.values() for t in v.values() if t and t.strip())
    if total == 0:
        raise RuntimeError(
            "形態素解析の結果がすべて空です。MeCab/fugashi の設定を確認してください。"
        )
    return parsed


def run_step_tfidf(parsed: Dict, log=print):
    """TF-IDF + LightGBM"""
    from src.data import authors_label
    from src.analysis import build_tfidf_dataset, train_tfidf_lgbm

    n = len(authors_label)
    vec, df_vec = build_tfidf_dataset(parsed)
    model, score = train_tfidf_lgbm(df_vec, target_num=n)
    log(f"  LightGBM テスト精度: {score['test_accuracy']:.4f}")
    save(vec, "tfidf.pkl")
    save(df_vec, "df_vec_tfidf.pkl")
    save(model, "lgbm_tfidf.pkl")
    return vec, df_vec, model


def run_step_embedding(corpus: Dict, log=print) -> Dict:
    """Sentence-BERT embedding"""
    from src.embedding import build_author_embedding_db

    log("  モデルをロード中...")
    author_vecs = build_author_embedding_db(corpus, show_progress=False)
    save(author_vecs, "author_vecs.pkl")
    log(f"  {len(author_vecs)} 作家分の embedding を計算完了")
    return author_vecs


def run_step_stylometry(corpus: Dict, log=print) -> Dict:
    """文体計量特徴"""
    from src.stylometry import extract_stylometric_features

    stylo: Dict = {}
    for idx, works in corpus.items():
        stylo[idx] = []
        for text in works.values():
            try:
                stylo[idx].append(extract_stylometric_features(text))
            except Exception as e:
                log(f"  WARNING: stylometry [{idx}]: {e}")
    save(stylo, "stylometry.pkl")
    return stylo


# ── 全データ一括ロード ────────────────────────────────────────────────────────

def load_all() -> tuple:
    """キャッシュからすべてのデータを読み込んで返す"""
    author_vecs = load("author_vecs.pkl")
    df_vec = load("df_vec_tfidf.pkl")
    lgbm = load("lgbm_tfidf.pkl")
    stylo = load("stylometry.pkl")
    tfidf_vec = load("tfidf.pkl")
    return author_vecs, df_vec, lgbm, stylo, tfidf_vec
