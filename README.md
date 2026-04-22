# 文豪文体評価アプリ (Buntai-Rating)

PDF・Word・テキストファイルまたは URL を入力すると、青空文庫から収集した文豪15名の作品と文体の類似度を分析するWebアプリです。

## 対応文豪（15名）

芥川龍之介 / 泉鏡花 / 江戸川乱歩 / 梶井基次郎 / 国木田独歩 / 坂口安吾 / 太宰治 / 谷崎潤一郎 / 徳冨蘆花 / 夏目漱石 / 二葉亭四迷 / 森鴎外 / 宮沢賢治 / 柳田国男 / 与謝野晶子

## 機能

| タブ | 内容 |
|------|------|
| 文体類似度スコア | Sentence-BERT コサイン類似度による文豪15名との類似度（100点満点） |
| 文体特徴分析 | 漢字率・語彙豊富度・品詞比率など文体計量特徴と作家比較 |
| コーパス可視化 | TF-IDF / Sentence-BERT embedding の UMAP・PCA 散布図・特徴量重要度 |
| モデル情報 | 使用モデル・手法の解説 |

## 使用モデル・手法

### Embeddingモデル
**[sonoisa/sentence-bert-base-ja-mean-tokens-v2](https://huggingface.co/sonoisa/sentence-bert-base-ja-mean-tokens-v2)**

日本語SentenceBERT (v2)。日本語NLI・STSタスクでファインチューニング済みであり、
従来の `cl-tohoku/bert-base-japanese-whole-word-masking` + mean-pooling より
文レベル表現の精度が高い。

### 分析手法

- **TF-IDF分析**: 作家ごとの特徴語抽出 + LightGBM 15クラス分類
- **Sentence-BERT 類似度評価**: 入力テキスト vs 各作家全作品のコサイン類似度
- **スタイロメトリー（追加）**: 漢字率・語彙豊富度・平均文長・品詞比率など文体計量特徴

## セットアップ

```bash
# 1. 依存ライブラリをインストール
pip install -r requirements.txt

# 2. 青空文庫からテキスト取得 + 前処理（初回のみ、数時間かかります）
python scripts/preprocess.py

# 3. Web アプリを起動
streamlit run app.py
```

## プロジェクト構成

```
.
├── app.py                   # Streamlit Webアプリ
├── scripts/
│   └── preprocess.py        # 前処理スクリプト（青空文庫取得・embedding計算）
├── src/
│   ├── data.py              # 文豪URLリスト・ラベル定義
│   ├── text_processing.py   # テキスト抽出・MeCab解析
│   ├── analysis.py          # TF-IDF・LightGBM・UMAP/PCA
│   ├── embedding.py         # Sentence-BERT embedding・類似度計算
│   └── stylometry.py        # 文体計量特徴抽出
├── data/                    # キャッシュファイル保存先（.pkl）
├── Buntai_Rating.ipynb      # 原版 Jupyter Notebook
└── requirements.txt
```

## ライセンス

青空文庫テキストは著作権フリー（著作者没後70年経過）です。