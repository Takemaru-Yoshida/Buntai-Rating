"""
文豪文体評価 Web アプリ (Streamlit)

起動方法:
    streamlit run app.py

前提: scripts/preprocess.py を実行してキャッシュファイルが data/ に存在すること
"""

import os
import pickle
import io
from statistics import mean

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import japanize_matplotlib  # noqa: F401
import plotly.graph_objects as go

from src.data import authors_label
from src.text_processing import (
    fetch_from_url,
    extract_from_pdf,
    extract_from_docx,
    extract_from_txt,
)
from src.embedding import encode_single, calculate_similarity, score_to_point
from src.stylometry import extract_stylometric_features, FEATURE_LABELS_JA
from src.analysis import plot_umap, plot_pca, plot_feature_importance

DATA_DIR = "data"
AUTHOR_NAMES = list(authors_label.keys())
N_AUTHORS = len(AUTHOR_NAMES)


# ── キャッシュロード ──────────────────────────────────────────────────────────

@st.cache_resource
def load_author_vecs():
    path = os.path.join(DATA_DIR, "author_vecs.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_tfidf_model():
    path = os.path.join(DATA_DIR, "lgbm_tfidf.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_df_vec_tfidf():
    path = os.path.join(DATA_DIR, "df_vec_tfidf.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_stylo_cache():
    path = os.path.join(DATA_DIR, "stylometry.pkl")
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        return pickle.load(f)


# ── ヘルパー ─────────────────────────────────────────────────────────────────

def _radar_chart(scores: list[float], labels: list[str], title: str) -> go.Figure:
    fig = go.Figure(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=labels + [labels[0]],
        fill="toself",
        name=title,
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title=title,
        height=500,
    )
    return fig


def _bar_chart(scores: list[float], labels: list[str], title: str) -> go.Figure:
    sorted_pairs = sorted(zip(scores, labels), reverse=True)
    s_scores, s_labels = zip(*sorted_pairs)
    colors = ["#e74c3c" if i == 0 else "#3498db" for i in range(len(s_scores))]
    fig = go.Figure(go.Bar(
        x=list(s_labels),
        y=list(s_scores),
        marker_color=colors,
        text=[f"{v:.0f}" for v in s_scores],
        textposition="auto",
    ))
    fig.update_layout(
        title=title,
        yaxis=dict(range=[0, 100], title="スコア"),
        height=420,
    )
    return fig


def _stylo_compare_chart(input_feat: dict, author_mean: dict, author_name: str) -> go.Figure:
    """入力テキストと選択作家の文体特徴を比較するバーチャート"""
    keys = [k for k in FEATURE_LABELS_JA if k in input_feat and k in author_mean]
    labels = [FEATURE_LABELS_JA[k] for k in keys]
    input_vals = [input_feat[k] for k in keys]
    author_vals = [author_mean[k] for k in keys]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="入力テキスト", x=labels, y=input_vals, marker_color="#e74c3c"))
    fig.add_trace(go.Bar(name=author_name, x=labels, y=author_vals, marker_color="#3498db"))
    fig.update_layout(
        barmode="group",
        title=f"文体特徴比較: 入力 vs {author_name}",
        height=450,
        xaxis_tickangle=-30,
    )
    return fig


# ── ページ設定 ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="文豪文体評価",
    page_icon="📖",
    layout="wide",
)

st.title("📖 文豪文体評価アプリ")
st.markdown(
    "PDF・Word・テキストファイルまたは URL を入力し、"
    "青空文庫の文豪15名との文体類似度を分析します。"
)

# ── サイドバー: 入力 ──────────────────────────────────────────────────────────

with st.sidebar:
    st.header("入力方法を選択")
    input_mode = st.radio("", ["ファイルアップロード", "URL入力"], index=0)

    raw_text = ""

    if input_mode == "ファイルアップロード":
        uploaded = st.file_uploader(
            "PDF / DOCX / TXT ファイルをアップロード",
            type=["pdf", "docx", "txt"],
        )
        if uploaded:
            ext = uploaded.name.rsplit(".", 1)[-1].lower()
            file_bytes = uploaded.read()
            with st.spinner("テキスト抽出中..."):
                try:
                    if ext == "pdf":
                        raw_text = extract_from_pdf(file_bytes)
                    elif ext == "docx":
                        raw_text = extract_from_docx(file_bytes)
                    else:
                        raw_text = extract_from_txt(file_bytes)
                    st.success(f"抽出完了（{len(raw_text)} 文字）")
                except Exception as e:
                    st.error(f"エラー: {e}")

    else:
        url_input = st.text_input("青空文庫などのURLを入力")
        if st.button("取得") and url_input:
            with st.spinner("テキスト取得中..."):
                try:
                    raw_text = fetch_from_url(url_input)
                    st.success(f"取得完了（{len(raw_text)} 文字）")
                except Exception as e:
                    st.error(f"エラー: {e}")

    st.divider()
    if raw_text:
        st.markdown("**テキストプレビュー**")
        st.text(raw_text[:500] + ("..." if len(raw_text) > 500 else ""))

# ── メインエリア ──────────────────────────────────────────────────────────────

author_vecs = load_author_vecs()
lgbm_model = load_tfidf_model()
df_vec_tfidf = load_df_vec_tfidf()
stylo_cache = load_stylo_cache()

if not raw_text:
    st.info("左サイドバーからテキストを入力または URL を指定してください。")

    # データ未準備の場合の案内
    if author_vecs is None:
        st.warning(
            "⚠️ キャッシュが未生成です。初回は以下を実行してください:\n\n"
            "```bash\npython scripts/preprocess.py\n```"
        )
    st.stop()

if author_vecs is None:
    st.error("キャッシュが見つかりません。`python scripts/preprocess.py` を実行してください。")
    st.stop()

# ── 分析実行 ──────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 文体類似度スコア",
    "🔬 文体特徴分析",
    "📈 コーパス可視化",
    "ℹ️ モデル情報",
])

# ─ Tab 1: 類似度スコア ────────────────────────────────────────────────────────
with tab1:
    st.subheader("文体類似度スコア（Sentence-BERT コサイン類似度）")
    with st.spinner("Sentence-BERT でエンコード中..."):
        query_vec = encode_single(raw_text[:3000])  # 先頭3000文字で代表
        sim_mean, sim_max = calculate_similarity(query_vec, author_vecs, N_AUTHORS)
        mean_pts = [score_to_point(s) for s in sim_mean]
        max_pts = [score_to_point(s) for s in sim_max]

    col_l, col_r = st.columns(2)
    with col_l:
        st.plotly_chart(
            _bar_chart(mean_pts, AUTHOR_NAMES, "平均類似度スコア（100点満点）"),
            use_container_width=True,
        )
    with col_r:
        st.plotly_chart(
            _radar_chart(mean_pts, AUTHOR_NAMES, "平均類似度レーダー"),
            use_container_width=True,
        )

    st.subheader("詳細スコア一覧")
    df_score = pd.DataFrame({
        "作家": AUTHOR_NAMES,
        "平均類似度（点）": mean_pts,
        "最高類似度（点）": max_pts,
    }).sort_values("平均類似度（点）", ascending=False).reset_index(drop=True)
    df_score.index += 1
    st.dataframe(df_score, use_container_width=True)

    top1 = df_score.iloc[0]["作家"]
    top1_score = df_score.iloc[0]["平均類似度（点）"]
    st.success(f"🏆 最も文体が近い文豪: **{top1}**（{top1_score} 点）")

# ─ Tab 2: 文体特徴分析 ────────────────────────────────────────────────────────
with tab2:
    st.subheader("入力テキストの文体計量特徴")
    with st.spinner("文体特徴を分析中..."):
        input_feat = extract_stylometric_features(raw_text)

    # 特徴表示
    feat_display = {FEATURE_LABELS_JA.get(k, k): round(v, 4) for k, v in input_feat.items()
                    if k != "char_count"}
    col1, col2, col3 = st.columns(3)
    feat_items = list(feat_display.items())
    for i, (label, val) in enumerate(feat_items):
        col = [col1, col2, col3][i % 3]
        col.metric(label, f"{val:.4f}")

    st.markdown(f"**総文字数**: {int(input_feat.get('char_count', 0)):,} 文字")

    # 作家との比較
    if stylo_cache:
        st.subheader("文体特徴の作家比較")
        selected_author = st.selectbox("比較する作家を選択", AUTHOR_NAMES)
        author_idx = authors_label[selected_author]

        # 選択作家の平均特徴
        works_feats = stylo_cache.get(author_idx, [])
        if works_feats:
            all_keys = [k for k in FEATURE_LABELS_JA if k != "char_count"]
            author_mean = {
                k: mean(f[k] for f in works_feats if k in f)
                for k in all_keys
                if any(k in f for f in works_feats)
            }
            st.plotly_chart(
                _stylo_compare_chart(input_feat, author_mean, selected_author),
                use_container_width=True,
            )

            # 類似スコア（ユークリッド距離ベース）
            common_keys = [k for k in all_keys if k in input_feat and k in author_mean]
            if common_keys:
                dist = np.linalg.norm(
                    np.array([input_feat[k] for k in common_keys])
                    - np.array([author_mean[k] for k in common_keys])
                )
                st.info(f"文体特徴ユークリッド距離 ({selected_author}): {dist:.4f}（小さいほど近い）")

# ─ Tab 3: コーパス可視化 ───────────────────────────────────────────────────────
with tab3:
    st.subheader("コーパスの分布可視化")
    if df_vec_tfidf is not None and lgbm_model is not None:
        X_all = df_vec_tfidf.drop(columns=["author"]).values
        y_all = df_vec_tfidf["author"].values

        col_a, col_b = st.columns(2)
        with col_a:
            with st.spinner("UMAP を計算中（初回は時間がかかります）..."):
                fig_umap = plot_umap(X_all, y_all, AUTHOR_NAMES, title="TF-IDF UMAP")
            st.pyplot(fig_umap)
            plt.close(fig_umap)

        with col_b:
            fig_pca = plot_pca(X_all, y_all, AUTHOR_NAMES, title="TF-IDF PCA")
            st.pyplot(fig_pca)
            plt.close(fig_pca)

        st.subheader("TF-IDF 特徴量重要度")
        feature_names_list = [str(c) for c in df_vec_tfidf.columns if c != "author"]
        fig_imp = plot_feature_importance(lgbm_model, feature_names_list, top_n=30)
        st.pyplot(fig_imp)
        plt.close(fig_imp)
    else:
        st.info("コーパス可視化には `python scripts/preprocess.py` の実行が必要です。")

    # Embedding UMAP
    if author_vecs:
        st.subheader("Sentence-BERT Embedding UMAP")
        emb_list, label_list = [], []
        for idx, vecs in author_vecs.items():
            emb_list.append(vecs)
            label_list.extend([idx] * len(vecs))
        X_emb = np.vstack(emb_list)
        y_emb = np.array(label_list)
        with st.spinner("Embedding UMAP を計算中..."):
            fig_emb_umap = plot_umap(X_emb, y_emb, AUTHOR_NAMES, title="Sentence-BERT UMAP")
        st.pyplot(fig_emb_umap)
        plt.close(fig_emb_umap)

# ─ Tab 4: モデル情報 ─────────────────────────────────────────────────────────
with tab4:
    st.subheader("使用モデル・手法")
    st.markdown("""
### Embeddingモデル
**[sonoisa/sentence-bert-base-ja-mean-tokens-v2](https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens-v2)**

日本語 SentenceBERT (v2)。日本語の自然言語推論・文類似度タスクでファインチューニングされており、
文レベルの意味・スタイル表現の質が高い。
従来の `cl-tohoku/bert-base-japanese-whole-word-masking` + 手動 mean-pooling より
文体類似度の精度が向上しています。

### 分析手法

| フェーズ | 手法 | 説明 |
|----------|------|------|
| **特徴分析** | TF-IDF (max 300語) | 作家ごとの頻出語彙を抽出 |
| **分類モデル** | LightGBM | TF-IDF特徴量による15クラス分類 |
| **次元削減** | UMAP / PCA | コーパス分布の可視化 |
| **類似度評価** | Sentence-BERT + コサイン類似度 | 入力テキストと各作家の作品ベクトルを比較 |
| **スタイロメトリー** (追加) | 文体計量分析 | 漢字率・語彙豊富度・文長・品詞比率など |

### スコア算出方法
- 入力テキストを Sentence-BERT でエンコードし、各作家の全作品ベクトルとコサイン類似度を計算
- **平均類似度**: 全作品との類似度の平均
- **最高類似度**: 全作品との類似度の最大値
- いずれも作家間で正規化（標準正規分布CDF）し、0〜100点に換算

### 文体計量特徴（スタイロメトリー）
| 特徴量 | 説明 |
|--------|------|
| 漢字率 | テキスト中の漢字の割合 |
| ひらがな率 | ひらがなの割合 |
| カタカナ率 | カタカナの割合 |
| 平均文長 | 1文あたりの平均文字数 |
| 語彙豊富度 | 異なり語数 / 総語数（TTR） |
| 句読点密度 | 句読点数 / 総文字数 |
| 1文あたり読点数 | 節の多さの代理指標 |
| 品詞比率 | 名詞・動詞・形容詞・副詞・助詞の割合 |
""")
