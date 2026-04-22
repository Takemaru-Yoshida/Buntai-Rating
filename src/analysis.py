"""TF-IDF・LightGBM・UMAP/PCA分析モジュール"""

from __future__ import annotations
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn import metrics
import lightgbm as lgb
import umap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401


def build_tfidf_dataset(
    dic: Dict[int, Dict[int, str]],
    max_features: int = 300,
) -> Tuple[TfidfVectorizer, pd.DataFrame]:
    """
    dic: {author_idx: {work_idx: parsed_text}}
    Returns vectorizer and DataFrame with columns ['author', feature...]
    """
    author_ids = [k for k, v in dic.items() for _ in v.values()]
    bodies = [vv for v in dic.values() for vv in v.values()]

    valid = [(a, b) for a, b in zip(author_ids, bodies) if b and b.strip()]
    if not valid:
        raise ValueError(
            "解析済みテキストが空です。parsed.pkl を削除して再セットアップしてください。"
        )
    author_ids_f, bodies_f = zip(*valid)

    df = pd.DataFrame({"author": list(author_ids_f), "body": list(bodies_f)})
    vec = TfidfVectorizer(max_features=max_features, norm="l2")
    X = vec.fit_transform(df["body"].values)
    df_vec = pd.concat(
        [pd.DataFrame({"author": df["author"].values}), pd.DataFrame(X.toarray())],
        axis=1,
    )
    return vec, df_vec


def train_tfidf_lgbm(
    df_vec: pd.DataFrame,
    target_num: int = 15,
    test_size: float = 0.3,
    val_ratio: float = 0.3,
    num_boost_round: int = 100,
) -> Tuple[lgb.Booster, Dict[str, float]]:
    """TF-IDF特徴量でLightGBM多クラス分類器を訓練する"""
    df_train, df_test = train_test_split(
        df_vec, test_size=test_size, shuffle=True, random_state=42,
        stratify=df_vec["author"],
    )
    X = df_train.drop(columns=["author"]).values
    y = df_train["author"].values
    X_tr, X_val, y_tr, y_val = train_test_split(X, y, test_size=val_ratio, random_state=0)

    lgb_tr = lgb.Dataset(X_tr, y_tr)
    lgb_val = lgb.Dataset(X_val, y_val, reference=lgb_tr)
    params = {
        "objective": "multiclass",
        "num_class": target_num,
        "metric": "multi_logloss",
        "verbosity": -1,
    }
    model = lgb.train(
        params, lgb_tr, valid_sets=[lgb_val],
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(period=-1)],
    )

    X_test = df_test.drop(columns=["author"]).values
    y_test = df_test["author"].values
    y_pred = np.argmax(model.predict(X_test, num_iteration=model.best_iteration), axis=1)
    score = {
        "test_accuracy": float(metrics.accuracy_score(y_test, y_pred)),
    }
    return model, score


def plot_feature_importance(
    model: lgb.Booster,
    feature_names: List[str],
    top_n: int = 30,
) -> plt.Figure:
    """LightGBMの特徴量重要度を棒グラフで返す"""
    importances = model.feature_importance()
    idx = np.argsort(importances)[::-1][:top_n]
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(
        [feature_names[i] for i in idx[::-1]],
        importances[idx[::-1]],
    )
    ax.set_title(f"特徴量重要度 Top {top_n}")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    return fig


def plot_umap(
    X: np.ndarray,
    y: np.ndarray,
    author_names: List[str],
    title: str = "UMAP",
    random_state: int = 0,
) -> plt.Figure:
    """UMAP次元削減の2D散布図を返す"""
    mapper = umap.UMAP(random_state=random_state)
    embedding = mapper.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for label, name in enumerate(author_names):
        mask = y == label
        ax.scatter(embedding[mask, 0], embedding[mask, 1], label=name, alpha=0.7, s=20)
    ax.legend(fontsize=7, loc="best")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_pca(
    X: np.ndarray,
    y: np.ndarray,
    author_names: List[str],
    title: str = "PCA",
) -> plt.Figure:
    """PCA 2D散布図を返す"""
    pca = PCA(n_components=2)
    X_r = pca.fit_transform(X)

    fig, ax = plt.subplots(figsize=(10, 8))
    for label, name in enumerate(author_names):
        mask = y == label
        ax.scatter(X_r[mask, 0], X_r[mask, 1], label=name, alpha=0.7, s=20)
    ax.legend(fontsize=7, loc="best")
    ax.set_title(f"{title} (explained: {pca.explained_variance_ratio_.sum():.1%})")
    plt.tight_layout()
    return fig
