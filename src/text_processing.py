"""テキスト抽出・前処理モジュール"""

import re
import io
import requests
from bs4 import BeautifulSoup
import MeCab


_tagger = None


def _get_tagger() -> MeCab.Tagger:
    global _tagger
    if _tagger is None:
        _tagger = MeCab.Tagger("-Owakati")
        _tagger.parse("")
    return _tagger


def extractbody(text: str) -> str:
    """青空文庫テキストからヘッダー・フッター・ルビ等を除去して本文を返す"""
    if re.search("-{5,}", text) and len(re.split("-{5,}", text)) >= 3:
        text = re.split("-{5,}", text)[2]
    else:
        text = "\n".join(text.split("\n")[2:])
    text = re.split("底本：", text)[0]
    text = re.sub("[|｜]", "", text)
    text = re.sub("《.+?》", "", text)
    text = re.sub("［＃.+?］", "", text)
    text = re.sub("\n\n", "\n", text)
    text = re.sub("\u3000", "", text)
    text = re.sub("\r", "", text)
    return text.strip()


def parsetext(text: str) -> str:
    """MeCabで形態素解析し、原形スペース区切りを返す"""
    tagger = _get_tagger()
    node = tagger.parseToNode(text)
    tokens = []
    while node:
        parts = node.feature.split(",")
        if len(parts) > 6 and parts[6] != "*":
            tokens.append(parts[6])
        elif node.surface:
            tokens.append(node.surface)
        node = node.next
    return " ".join(tokens)


def fetch_from_url(url: str) -> str:
    """URLから青空文庫テキストを取得して本文文字列を返す"""
    response = requests.get(url, timeout=30)
    response.encoding = response.apparent_encoding
    soup = BeautifulSoup(response.content, "html.parser")
    div = soup.find("div", class_="main_text")
    if div is None:
        raise ValueError(f"本文が見つかりません: {url}")
    for tag in div.find_all(["rp", "rt"]):
        tag.decompose()
    return extractbody(div.get_text())


def extract_from_pdf(file_bytes: bytes) -> str:
    """PDFバイト列からテキストを抽出する"""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return extractbody("\n".join(pages))
    except ImportError:
        import PyPDF2
        reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
        pages = [page.extract_text() or "" for page in reader.pages]
        return extractbody("\n".join(pages))


def extract_from_docx(file_bytes: bytes) -> str:
    """DOCXバイト列からテキストを抽出する"""
    import docx2txt
    import tempfile, os
    with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        text = docx2txt.process(tmp_path)
    finally:
        os.unlink(tmp_path)
    return extractbody(text)


def extract_from_txt(file_bytes: bytes) -> str:
    """TXTバイト列からテキストを抽出する"""
    text = file_bytes.decode("utf-8", errors="replace")
    return extractbody(text)
