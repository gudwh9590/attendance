# embedding.py 전체 코드
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

#현재 실행 중인 파이썬 파일이 있는 폴더의 절대 경로
BASE_DIR = Path(__file__).resolve().parent 

# 1. pdf 를 document 객체로 변환
documents = []

for pdf_file in BASE_DIR.glob("*.pdf"): 
    loader = PyPDFLoader(str(pdf_file))
    documents.extend(loader.load())

print("페이지수:",len(documents))

# 2. 문서 분할 (단위기준 표가 한 덩어리로 묶이도록 size와 overlap 증가)
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200 
)

docs = splitter.split_documents(documents)
print("청크 갯수",len(docs))

# 3.임베딩 작업
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
)

# 4.크로마 DB 생성
DB_PATH = BASE_DIR / "chroma_db"

db = Chroma.from_documents(
    documents=docs,
    embedding=embedding,
    persist_directory=str(DB_PATH)
)
print("Vector DB 저장 완료")