"""
前処理スクリプト: 青空文庫からテキストを取得し、
TF-IDF特徴量・LightGBMモデル・Sentence-BERTembeddingをキャッシュとして保存する。

実行方法:
    python scripts/preprocess.py

生成ファイル:
    data/corpus.pkl        - {author_idx: {work_idx: raw_text}} の辞書
    data/parsed.pkl        - {author_idx: {work_idx: parsed_text (MeCab)}} の辞書
    data/tfidf.pkl         - TfidfVectorizer
    data/df_vec_tfidf.pkl  - TF-IDFベクトルのDataFrame
    data/lgbm_tfidf.pkl    - TF-IDF LightGBM モデル
    data/author_vecs.pkl   - {author_idx: ndarray} Sentence-BERT embeddings
    data/stylometry.pkl    - {author_idx: List[Dict]} stylometric features per work
"""

import os
import sys
import pickle
import time

from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.data import urls, authors_label
from src.text_processing import fetch_from_url, parsetext
from src.analysis import build_tfidf_dataset, train_tfidf_lgbm
from src.embedding import build_author_embedding_db
from src.stylometry import extract_stylometric_features

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
os.makedirs(DATA_DIR, exist_ok=True)


def _save(obj, filename: str):
    path = os.path.join(DATA_DIR, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  saved -> {path}")


def _load(filename: str):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        with open(path, "rb") as f:
            return pickle.load(f)
    return None


def step1_scrape():
    """Step 1: 青空文庫から本文テキストを取得"""
    corpus = _load("corpus.pkl") or {}
    for idx, (author, work_urls) in enumerate(tqdm(urls.items(), desc="authors")):
        if idx in corpus:
            print(f"  skip {author} (cached)")
            continue
        corpus[idx] = {}
        for work_idx, url in enumerate(tqdm(work_urls, desc=author, leave=False)):
            try:
                text = fetch_from_url(url)
                corpus[idx][work_idx] = text
                time.sleep(0.5)  # Aozora Bunko への負荷軽減
            except Exception as e:
                print(f"    ERROR {url}: {e}")
        _save(corpus, "corpus.pkl")
    return corpus


def step2_parse(corpus):
    """Step 2: MeCab形態素解析"""
    parsed = _load("parsed.pkl") or {}
    for idx, works in tqdm(corpus.items(), desc="parsing"):
        if idx in parsed:
            continue
        parsed[idx] = {}
        for work_idx, text in works.items():
            try:
                parsed[idx][work_idx] = parsetext(text)
            except Exception as e:
                print(f"    parse error [{idx}][{work_idx}]: {e}")
    _save(parsed, "parsed.pkl")
    return parsed


def step3_tfidf(parsed):
    """Step 3: TF-IDFベクトル化 + LightGBM"""
    vec, df_vec = build_tfidf_dataset(parsed, max_features=300)
    author_names = list(authors_label.keys())
    target_num = len(author_names)
    model, score = train_tfidf_lgbm(df_vec, target_num=target_num)
    print(f"  TF-IDF LightGBM test accuracy: {score['test_accuracy']:.4f}")
    _save(vec, "tfidf.pkl")
    _save(df_vec, "df_vec_tfidf.pkl")
    _save(model, "lgbm_tfidf.pkl")
    return vec, df_vec, model


def step4_embedding(corpus):
    """Step 4: Sentence-BERT embedding"""
    print("  Loading Sentence-BERT model...")
    author_vecs = build_author_embedding_db(corpus, show_progress=True)
    _save(author_vecs, "author_vecs.pkl")
    return author_vecs


def step5_stylometry(corpus):
    """Step 5: 文体計量特徴"""
    stylo = {}
    for idx, works in tqdm(corpus.items(), desc="stylometry"):
        stylo[idx] = []
        for text in works.values():
            try:
                stylo[idx].append(extract_stylometric_features(text))
            except Exception as e:
                print(f"    stylometry error [{idx}]: {e}")
    _save(stylo, "stylometry.pkl")
    return stylo


def main():
    print("=== Step 1: Scraping Aozora Bunko ===")
    corpus = step1_scrape()

    print("\n=== Step 2: MeCab Parsing ===")
    parsed = step2_parse(corpus)

    print("\n=== Step 3: TF-IDF + LightGBM ===")
    step3_tfidf(parsed)

    print("\n=== Step 4: Sentence-BERT Embedding ===")
    step4_embedding(corpus)

    print("\n=== Step 5: Stylometric Analysis ===")
    step5_stylometry(corpus)

    print("\nDone! All cache files saved to data/")


if __name__ == "__main__":
    main()
