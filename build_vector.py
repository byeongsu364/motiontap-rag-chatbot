import os
import re
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPowerPointLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()

DATA_DIR = "./data"
VECTOR_DIR = "./vectorstore/faiss_index"


def get_page_number(doc):
    page = doc.metadata.get("page")
    page_number = doc.metadata.get("page_number")

    if page is not None:
        try:
            return int(page) + 1
        except Exception:
            return page

    if page_number is not None:
        try:
            return int(page_number)
        except Exception:
            return page_number

    return "페이지 정보 없음"


def get_section_name(text):
    patterns = [
        r"\d+-\d+\.\s*[^\n]+",
        r"\d+\s*-\s*\d+\.\s*[^\n]+",
        r"PART\s*\d+\.?\s*[^\n]*",
        r"[0-9]+\.\s*[^\n]+",
        r"모션탭\s*주요\s*기능",
        r"제품\s*사양",
        r"제품\s*주의\s*사항",
        r"어플설치\s*및\s*회원가입",
        r"제품\s*충전\s*및\s*영상가이드",
        r"화면기본\s*구성",
        r"동작방식\s*설명",
        r"프로그램\s*활용\s*예제",
        r"제품\s*사용\s*프로세스",
        r"기본\s*오류\s*대처",
        r"터치\s*센서\s*오류\s*대처"
    ]

    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(0).strip()

    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for line in lines[:8]:
        if len(line) <= 35 and any(keyword in line for keyword in ["모션탭", "제품", "화면", "동작", "프로그램", "오류", "센서", "회원가입", "충전"]):
            return line

    return "섹션 정보 없음"


documents = []

for file_name in os.listdir(DATA_DIR):
    file_path = os.path.join(DATA_DIR, file_name)

    if file_name.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
        docs = loader.load()

    elif file_name.endswith(".pptx"):
        loader = UnstructuredPowerPointLoader(file_path, mode="elements")
        docs = loader.load()

    else:
        continue

    for doc in docs:
        doc.metadata["source"] = file_name
        doc.metadata["page"] = get_page_number(doc)
        doc.metadata["section"] = get_section_name(doc.page_content)

    documents.extend(docs)

print(f"불러온 문서 수: {len(documents)}")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

splits = text_splitter.split_documents(documents)

for split in splits:
    if not split.metadata.get("section") or split.metadata.get("section") == "섹션 정보 없음":
        split.metadata["section"] = get_section_name(split.page_content)

print(f"청크 수: {len(splits)}")

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = FAISS.from_documents(splits, embeddings)
vectorstore.save_local(VECTOR_DIR)

print("FAISS 벡터 DB 저장 완료")
print("메타데이터 저장 완료: source, page, section")