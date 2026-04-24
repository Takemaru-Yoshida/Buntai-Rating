"""
データキャッシュ管理モジュール

読み込み優先順位:
  1. ./data/             ← リポジトリにコミットされた事前ビルド済みファイル
  2. /tmp/buntai_cache/  ← セットアップ実行時の書き込み先（Streamlit Cloud 等）
  3. ./data/             ← ローカル開発時の書き込み先（/tmp が使えない場合）

Streamlit Community Cloud では data/ にコミット済みファイルがあれば
セットアップをスキップして即座にアプリを起動する。
"""

from __future__ import annotations

import os
import pickle
import time
from typing import Any, Dict, Optional

# data/ ディレクトリ（リポジトリ内、事前ビルド済みファイルの置き場）
DATA_DIR: str = os.path.join(os.path.dirname(__file__), "..", "data")

# 書き込み先（Streamlit Cloud では /tmp、ローカルでは data/）
CACHE_DIR: str = os.environ.get(
    "BUNTAI_CACHE_DIR",
    "/tmp/buntai_cache" if os.access("/tmp", os.W_OK) else DATA_DIR,
)

REQUIRED_FILES = [
    "author_vecs.pkl",
    "df_vec_tfidf.pkl",
    "lgbm_tfidf.pkl",
    "tfidf.pkl",
    "stylometry.pkl",
    "viz.pkl",
]


def data_path(filename: str) -> str:
    return os.path.join(DATA_DIR, filename)


def cache_path(filename: str) -> str:
    return os.path.join(CACHE_DIR, filename)


def is_step_done(filename: str) -> bool:
    """data/ または CACHE_DIR にファイルが存在するか確認する"""
    return os.path.exists(data_path(filename)) or os.path.exists(cache_path(filename))


def is_setup_complete() -> bool:
    return all(is_step_done(f) for f in REQUIRED_FILES)


def save(obj: Any, filename: str) -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(cache_path(filename), "wb") as f:
        pickle.dump(obj, f)


def load(filename: str) -> Optional[Any]:
    """data/ を優先して読み込む（事前ビルド済みファイルが存在する場合）"""
    for path in [data_path(filename), cache_path(filename)]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                return pickle.load(f)
    return None


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

def run_step_viz(df_vec_tfidf, author_vecs: Dict, log=print) -> Dict:
    """UMAP / PCA 座標を事前計算して保存する"""
    import numpy as np
    from src.analysis import compute_umap, compute_pca

    X_tfidf = df_vec_tfidf.drop(columns=["author"]).values
    y_tfidf = df_vec_tfidf["author"].values

    log("  TF-IDF UMAP を計算中...")
    tfidf_umap = compute_umap(X_tfidf)
    log("  TF-IDF PCA を計算中...")
    tfidf_pca, tfidf_pca_exp = compute_pca(X_tfidf)

    emb_list, label_list = [], []
    for idx, vecs in author_vecs.items():
        emb_list.append(vecs)
        label_list.extend([idx] * len(vecs))
    X_emb = np.vstack(emb_list)
    y_emb = np.array(label_list)

    log("  Embedding UMAP を計算中...")
    emb_umap = compute_umap(X_emb)

    viz = {
        "tfidf_umap": tfidf_umap,
        "tfidf_pca": tfidf_pca,
        "tfidf_pca_exp": tfidf_pca_exp,
        "tfidf_labels": y_tfidf,
        "emb_umap": emb_umap,
        "emb_labels": y_emb,
    }
    save(viz, "viz.pkl")
    log("  可視化データを保存しました")
    return viz


def load_all() -> tuple:
    """キャッシュからすべてのデータを読み込んで返す"""
    author_vecs = load("author_vecs.pkl")
    df_vec = load("df_vec_tfidf.pkl")
    lgbm = load("lgbm_tfidf.pkl")
    stylo = load("stylometry.pkl")
    tfidf_vec = load("tfidf.pkl")
    viz = load("viz.pkl")
    return author_vecs, df_vec, lgbm, stylo, tfidf_vec, viz
