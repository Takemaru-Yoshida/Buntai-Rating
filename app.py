"""
文豪文体評価 Web アプリ (Streamlit)

起動方法:
    streamlit run app.py

初回起動時は青空文庫からのスクレイピングと embedding 計算を自動実行します。
"""

import os
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
from src.analysis import scatter_2d, plot_feature_importance
import src.cache as cache

AUTHOR_NAMES = list(authors_label.keys())
N_AUTHORS = len(AUTHOR_NAMES)

# ── ページ設定 ─────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="文豪文体評価",
    page_icon="📖",
    layout="wide",
)

# ── インメモリキャッシュ（@st.cache_resource）────────────────────────────────

@st.cache_resource(show_spinner=False)
def _load_data_from_disk():
    """ディスクキャッシュから全データをメモリに読み込む（セッションをまたいで保持）"""
    return cache.load_all()


# ── セットアップページ ────────────────────────────────────────────────────────

def _run_setup_ui():
    """初回セットアップを段階的に実行し、st.status で進捗を表示する"""

    st.title("📖 文豪文体評価アプリ — 初回セットアップ")
    st.info(
        "初回のみ、青空文庫からテキストを取得してモデルを構築します。\n\n"
        "⏱ 目安時間:\n"
        "- スクレイピング: 約2〜3時間（15作家 × 100作品 × 0.5秒）\n"
        "- 形態素解析: 約10分\n"
        "- Sentence-BERT encoding: 約30分\n\n"
        "処理はサーバー側で実行され、完了後は再起動しない限り再実行不要です。"
    )

    if st.button("🚀 セットアップを開始", type="primary", use_container_width=True):
        _do_setup()
        st.success("✅ セットアップ完了！ページを更新して分析を開始してください。")
        st.balloons()
        if st.button("ページを更新", type="secondary"):
            st.rerun()


def _do_setup():
    """各ステップをst.statusで表示しながら実行する"""

    # ステップ 1: スクレイピング
    if not cache.is_step_done("corpus.pkl"):
        with st.status("Step 1/5 — 青空文庫からテキスト取得中...", expanded=True) as s:
            logs = []

            def log(msg):
                logs.append(msg)
                s.write(msg)

            corpus = cache.run_step_scrape(log=log)
            s.update(label=f"Step 1/5 完了 — {sum(len(v) for v in corpus.values())} 作品取得", state="complete")
    else:
        st.success("Step 1/5 — スクレイピング: キャッシュ済み ✓")
        corpus = cache.load("corpus.pkl")

    # ステップ 2: 形態素解析
    if not cache.is_step_done("parsed.pkl"):
        with st.status("Step 2/5 — 形態素解析中...", expanded=False) as s:
            parsed = cache.run_step_parse(corpus, log=s.write)
            s.update(label="Step 2/5 完了 — 形態素解析", state="complete")
    else:
        st.success("Step 2/5 — 形態素解析: キャッシュ済み ✓")
        parsed = cache.load("parsed.pkl")

    # ステップ 3: TF-IDF + LightGBM
    if not cache.is_step_done("lgbm_tfidf.pkl"):
        with st.status("Step 3/5 — TF-IDF + LightGBM 学習中...", expanded=False) as s:
            cache.run_step_tfidf(parsed, log=s.write)
            s.update(label="Step 3/5 完了 — TF-IDF + LightGBM", state="complete")
    else:
        st.success("Step 3/5 — TF-IDF + LightGBM: キャッシュ済み ✓")

    # ステップ 4: Sentence-BERT embedding
    if not cache.is_step_done("author_vecs.pkl"):
        with st.status("Step 4/5 — Sentence-BERT embedding 計算中...", expanded=False) as s:
            cache.run_step_embedding(corpus, log=s.write)
            s.update(label="Step 4/5 完了 — Sentence-BERT embedding", state="complete")
    else:
        st.success("Step 4/5 — Sentence-BERT embedding: キャッシュ済み ✓")

    # ステップ 5: スタイロメトリー
    if not cache.is_step_done("stylometry.pkl"):
        with st.status("Step 5/5 — 文体計量特徴を計算中...", expanded=False) as s:
            cache.run_step_stylometry(corpus, log=s.write)
            s.update(label="Step 5/5 完了 — 文体計量特徴", state="complete")
    else:
        st.success("Step 5/5 — 文体計量特徴: キャッシュ済み ✓")


# ── ヘルパー（チャート）──────────────────────────────────────────────────────

def _radar_chart(scores, labels, title):
    fig = go.Figure(go.Scatterpolar(
        r=scores + [scores[0]],
        theta=labels + [labels[0]],
        fill="toself",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=False,
        title=title,
        height=500,
    )
    return fig


def _bar_chart(scores, labels, title):
    pairs = sorted(zip(scores, labels), reverse=True)
    s_scores, s_labels = zip(*pairs)
    colors = ["#e74c3c" if i == 0 else "#3498db" for i in range(len(s_scores))]
    fig = go.Figure(go.Bar(
        x=list(s_labels),
        y=list(s_scores),
        marker_color=colors,
        text=[f"{v:.0f}" for v in s_scores],
        textposition="auto",
    ))
    fig.update_layout(title=title, yaxis=dict(range=[0, 100], title="スコア"), height=420)
    return fig


def _stylo_compare_chart(input_feat, author_mean, author_name):
    keys = [k for k in FEATURE_LABELS_JA if k in input_feat and k in author_mean]
    labels = [FEATURE_LABELS_JA[k] for k in keys]
    fig = go.Figure()
    fig.add_trace(go.Bar(name="入力テキスト", x=labels,
                         y=[input_feat[k] for k in keys], marker_color="#e74c3c"))
    fig.add_trace(go.Bar(name=author_name, x=labels,
                         y=[author_mean[k] for k in keys], marker_color="#3498db"))
    fig.update_layout(barmode="group",
                      title=f"文体特徴比較: 入力 vs {author_name}",
                      height=450, xaxis_tickangle=-30)
    return fig


# ── メインアプリ ──────────────────────────────────────────────────────────────

def _main_app(author_vecs, df_vec_tfidf, lgbm_model, stylo_cache, tfidf_vec, viz):

    st.title("📖 文豪文体評価アプリ")
    st.markdown(
        "PDF・Word・テキストファイルまたは URL を入力し、"
        "青空文庫の文豪15名との文体類似度を分析します。"
    )

    # ── サイドバー ────────────────────────────────────────────────────────────
    with st.sidebar:
        st.header("入力方法を選択")
        input_mode = st.radio("", ["ファイルアップロード", "URL入力"], index=0)
        raw_text = ""

        if input_mode == "ファイルアップロード":
            uploaded = st.file_uploader(
                "PDF / DOCX / TXT をアップロード",
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

        if raw_text:
            st.divider()
            st.markdown("**テキストプレビュー**")
            st.text(raw_text[:500] + ("..." if len(raw_text) > 500 else ""))

    if not raw_text:
        st.info("左サイドバーからテキストを入力または URL を指定してください。")
        st.stop()

    # ── タブ ─────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 文体類似度スコア",
        "🔬 文体特徴分析",
        "📈 コーパス可視化",
        "ℹ️ モデル情報",
    ])

    # Tab 1: 類似度スコア
    with tab1:
        st.subheader("文体類似度スコア（Sentence-BERT コサイン類似度）")
        with st.spinner("Sentence-BERT でエンコード中..."):
            query_vec = encode_single(raw_text[:3000])
            sim_mean, sim_max = calculate_similarity(query_vec, author_vecs, N_AUTHORS)
            mean_pts = [score_to_point(s) for s in sim_mean]
            max_pts = [score_to_point(s) for s in sim_max]

        col_l, col_r = st.columns(2)
        with col_l:
            st.plotly_chart(_bar_chart(mean_pts, AUTHOR_NAMES, "平均類似度スコア（100点満点）"),
                            use_container_width=True)
        with col_r:
            st.plotly_chart(_radar_chart(mean_pts, AUTHOR_NAMES, "平均類似度レーダー"),
                            use_container_width=True)

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

    # Tab 2: 文体特徴分析
    with tab2:
        st.subheader("入力テキストの文体計量特徴")
        with st.spinner("文体特徴を分析中..."):
            input_feat = extract_stylometric_features(raw_text)

        feat_display = {FEATURE_LABELS_JA.get(k, k): round(v, 4)
                        for k, v in input_feat.items() if k != "char_count"}
        col1, col2, col3 = st.columns(3)
        for i, (label, val) in enumerate(feat_display.items()):
            [col1, col2, col3][i % 3].metric(label, f"{val:.4f}")
        st.markdown(f"**総文字数**: {int(input_feat.get('char_count', 0)):,} 文字")

        if stylo_cache:
            st.subheader("文体特徴の作家比較")
            selected_author = st.selectbox("比較する作家を選択", AUTHOR_NAMES)
            author_idx = authors_label[selected_author]
            works_feats = stylo_cache.get(author_idx, [])
            if works_feats:
                all_keys = [k for k in FEATURE_LABELS_JA if k != "char_count"]
                author_mean = {
                    k: mean(f[k] for f in works_feats if k in f)
                    for k in all_keys if any(k in f for f in works_feats)
                }
                st.plotly_chart(_stylo_compare_chart(input_feat, author_mean, selected_author),
                                use_container_width=True)
                common = [k for k in all_keys if k in input_feat and k in author_mean]
                if common:
                    a = np.array([input_feat[k] for k in common])
                    b = np.array([author_mean[k] for k in common])
                    dist = np.linalg.norm(a - b)
                    st.metric("ユークリッド距離（小さいほど近い）", f"{dist:.4f}")

    # Tab 3: コーパス可視化
    with tab3:
        st.subheader("コーパスの分布可視化")
        if viz is not None:
            col_a, col_b = st.columns(2)
            with col_a:
                fig_umap = scatter_2d(
                    viz["tfidf_umap"], viz["tfidf_labels"], AUTHOR_NAMES, "TF-IDF UMAP"
                )
                st.pyplot(fig_umap)
                plt.close(fig_umap)
            with col_b:
                fig_pca = scatter_2d(
                    viz["tfidf_pca"], viz["tfidf_labels"], AUTHOR_NAMES,
                    "TF-IDF PCA", viz["tfidf_pca_exp"]
                )
                st.pyplot(fig_pca)
                plt.close(fig_pca)

            if lgbm_model is not None:
                st.subheader("TF-IDF 特徴量重要度")
                feature_names_list = (
                    list(tfidf_vec.get_feature_names_out())
                    if tfidf_vec is not None
                    else [str(c) for c in df_vec_tfidf.columns if c != "author"]
                )
                fig_imp = plot_feature_importance(lgbm_model, feature_names_list, top_n=30)
                st.pyplot(fig_imp)
                plt.close(fig_imp)

            st.subheader("Embedding UMAP")
            fig_emb = scatter_2d(
                viz["emb_umap"], viz["emb_labels"], AUTHOR_NAMES, "multilingual-e5-base UMAP"
            )
            st.pyplot(fig_emb)
            plt.close(fig_emb)

    # Tab 4: モデル情報
    with tab4:
        st.subheader("使用モデル・手法")
        st.markdown("""
### Embedding モデル
**[intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base)**

XLM-RoBERTa ベースの多言語埋め込みモデル（SentencePiece トークナイザー）。
MeCab などの形態素解析器に依存せず、日本語を含む100言語以上に対応する。
mean-pooling + L2 正規化により文書ベクトルを生成し、コサイン類似度で文体比較を行う。

### 形態素解析
**[fugashi](https://github.com/polm/fugashi) + [ipadic](https://github.com/polm/ipadic-py)**

TF-IDF および文体計量分析の前処理に使用。
システムの libmecab に依存しない純 Python 実装で、
Streamlit Community Cloud などのクラウド環境でも動作する。

### 分析手法

| フェーズ | 手法 | 説明 |
|----------|------|------|
| 特徴抽出 | TF-IDF（最大300語） | 形態素解析後の原形で作家ごとの頻出語彙を抽出 |
| 分類モデル | LightGBM | TF-IDF 特徴量による15クラス分類 |
| 次元削減 | UMAP / PCA（事前計算済み） | コーパス分布の可視化（起動時に即表示） |
| 類似度評価 | multilingual-e5-base + コサイン類似度 | 入力テキストと各作家の作品ベクトルを比較 |
| スタイロメトリー | 文体計量分析 | 漢字率・語彙豊富度・文長・品詞比率など13指標 |

### スコア算出方法
各作家の全作品との類似度を標準正規分布 CDF で正規化し 0〜100 点に換算。
平均類似度スコア（全作品との平均）と最高類似度スコア（最も近い作品との類似度）を提供。
""")


# ── エントリーポイント ────────────────────────────────────────────────────────

if not cache.is_setup_complete():
    _run_setup_ui()
else:
    with st.spinner("データを読み込んでいます..."):
        author_vecs, df_vec_tfidf, lgbm_model, stylo_cache, tfidf_vec, viz = _load_data_from_disk()
    _main_app(author_vecs, df_vec_tfidf, lgbm_model, stylo_cache, tfidf_vec, viz)
