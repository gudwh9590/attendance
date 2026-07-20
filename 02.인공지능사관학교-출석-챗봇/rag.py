# rag.py 전체 코드
from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
import sys

# 현재 파일 기준으로 4단계 위 폴더를 import 경로에 추가
sys.path.append(str(Path(__file__).resolve().parent.parent.parent.parent))

from llm_loader import init_custom_llm
llm = init_custom_llm()

# 이미 만들어진 크로마 DB 객체 생성
embedding = HuggingFaceEmbeddings(
    model_name="BAAI/bge-m3",
)

# 현재 rag.py가 있는 폴더
BASE_DIR = Path(__file__).resolve().parent

# chroma_db 폴더
DB_PATH = BASE_DIR / "chroma_db"

db = Chroma(
    embedding_function = embedding,
    persist_directory = str(DB_PATH)
)

# 검색기 (k값을 5로 늘려 단위일수 표가 검색 결과에 포함될 확률 극대화)
retriever = db.as_retriever(
    search_kwargs={"k": 5}
)

# 프롬프트 만들고 llm 연결 준비
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 프롬프트를 매우 구체적으로 통제
prompt = ChatPromptTemplate.from_template(
"""
당신은 광주 인공지능사관학교의 출결 관리 규정을 토대로 출결률을 알려주는 전문 AI 챗봇입니다.

[중요 지침]
1. 출석률이나 평일 일수를 계산할 때 절대 일반적인 기준(예: 20일)을 임의로 사용하지 마세요.
2. 질문에 해당하는 월이나 차수(1차~8차)를 파악한 뒤, 제공된 문서 내 '[월별 단위일수(평일) 및 출석률 계산 기준]' 또는 '단위기간 및 지급 시기' 표를 무조건 확인하세요.
3. 문서에 명시된 해당 차수의 '평일 단위일수'를 그대로 가져와서 계산하고 답변해야 합니다. (예: 2차는 평일 21일)
4. 문서에 없는 내용은 절대 지어내지 마세요.
5. 공결(공가)은 이미 출석한 것으로 취급하므로, 전체 출석일수를 계산할 때 숫자를 더하거나 빼지 말고 무시하세요. 출석일수는 오직 [해당 월의 평일 단위일수 - 최종 결석일수(지각/조퇴로 인한 차감 포함)] 공식으로만 계산하세요.
6. 지각, 조퇴, 외출은 발생 횟수를 모두 더한 뒤, 합산된 횟수가 3회가 될 때마다 결석 1회로 바꾸어 계산하세요. (예: 지각 2회 + 조퇴 1회 = 총 3회이므로 결석 1회로 추가 처리)
7. 최종 결석일수를 구할 때는 반드시 [실제 결석 횟수 + ((지각 횟수 + 조퇴 횟수 + 외출 횟수)를 3으로 나눈 몫)] 공식을 머릿속으로 계산한 뒤에, 그 과정을 사용자에게 단계별로 설명해 주세요.
문서

{context}

질문

{question}
"""
)

chain = prompt | llm | StrOutputParser()

# 관련문서 검색후 llm 연결
def format_docs(docs):
    result = ""
    for doc in docs:
        result = result + doc.page_content
        result = result + "\n\n"
    return result

def ask(question):
    # 1. 관련 문서 검색
    docs = retriever.invoke(question)
    
    # 2. 문자열로 변환
    context = format_docs(docs)
    
    # 3. Chain 실행
    answer = chain.invoke(
        {
        "context":context,
        "question": question
        }
    )

    return answer