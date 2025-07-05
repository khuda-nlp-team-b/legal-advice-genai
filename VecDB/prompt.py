import os
import argparse
import jinja2
import time
import utils as u                             # ← 추가
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_community.llms import HuggingFacePipeline
from dotenv import load_dotenv
load_dotenv()

# ────────────────── 1) 경로·환경 ──────────────────
BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(BASE_PATH, "prompts")

# ↓↓↓  db_test/<여기에 db 입력>  부분만 바꿔주세요 ↓↓↓
DB_SUBDIR = "LAW_RAG_500_75"
#DB_DIR    = os.path.join(BASE_PATH, "db_test", DB_SUBDIR)
# ↑↑↑  db_test/<여기에 db 입력>  부분만 바꿔주세요 ↑↑↑

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# ────────────────── 2) DB 접속 정보 ──────────────────
# (create_db.retrieve_db 호출을 위해 환경변수로 설정)
HOST        = os.getenv("DB_HOST")
PORT        = int(os.getenv("DB_PORT", 3306))
USER        = os.getenv("DB_USER")
PASSWORD    = os.getenv("DB_PASSWORD")
DB_NAME     = os.getenv("DB_NAME")
#BASE_DB_DIR = os.path.join(BASE_PATH, "db_test")   # create_db.py의 base_db_dir 인자

# ────────────────── 3) Jinja2 템플릿 로드 ──────────────────
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(PROMPT_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
#query_tpl  = env.get_template("query_rewrite.j2")  # 템플릿 파일명이 정확한지 확인하세요 :contentReference[oaicite:0]{index=0}
answer_tpl = env.get_template("answer_synth.j2")

def get_llm():
    return ChatOpenAI(api_key=OPENAI_KEY, model="gpt-4o-mini", temperature=0.5)

def rewrite_query(user_query: str) -> str:
    prompt = query_tpl.render(user_query=user_query)
    llm = get_llm()
    print("🔄 질의 재작성(LLM) …", end=" ")
    start = time.perf_counter()
    resp = llm.invoke(prompt)
    print(f"✔ ({time.perf_counter()-start:.1f}s)")
    return resp.content.strip() if hasattr(resp, "content") else resp.strip()

def run_rag(user_query: str, k: int = 10) -> str:
    # 1) 검색어 재작성
    #search_query = rewrite_query(user_query)
    #print("   ↪ 검색어:", search_query)

    # 2) MySQL+Chroma 통합 전문 조회 (create_db.retrieve_db 호출)
    print("🔍 MySQL 전문 조회 중…")
    results = u.retrieve_db(
        user_query,
        HOST, PORT, USER, PASSWORD, DB_NAME,k=k
    )

    # 3) 검색 결과 처리 및 템플릿 적용
    if not results or len(results) == 0:
        return "유사 판례를 찾지 못했습니다."

    # 가장 유사한 판례 1건만 활용 (확장 가능)
    top = results[0]
    context = f"{top['유사문단']} [ref:{top['판례일련번호']}]"
    full_document = top['전문']

    answer = answer_tpl.render(
        context=context,
        full_document=full_document,
        user_query=user_query
    )
    return answer


def main():

    user_query = input("💬 질문을 입력하세요: ").strip()

    answer = run_rag(user_query)
    
    print("\n📌 최종 요약\n", answer)
    
    return answer

if __name__ == "__main__":
    main()