import os
import argparse
import jinja2
import time
import utils as u  # 기존: create_db as cdb
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from dotenv import load_dotenv
load_dotenv()

# ──────────── 1) 경로·환경 ────────────
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(BASE_PATH, "prompts")
DB_SUBDIR = "LAW_RAG_500_75"
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
BASE_DB_DIR = os.path.join(BASE_PATH, "db_test")

# ──────────── 2) DB 접속 정보 ────────────
HOST = os.getenv("DB_HOST")
PORT = int(os.getenv("DB_PORT", 3306))
USER = os.getenv("DB_USER")
PASSWORD = os.getenv("DB_PASSWORD")
DB_NAME = os.getenv("DB_NAME")

# ──────────── 3) Jinja2 템플릿 로드 ────────────
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(PROMPT_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)
query_tpl  = env.get_template("query_rewrite.j2")
answer_tpl = env.get_template("answer_synth.j2")

def get_llm():
    return ChatOpenAI(api_key=OPENAI_KEY, model_name="gpt-3.5-turbo", temperature=1)

def run_rag(user_query: str, k: int = 5) -> str:
    # 1) 벡터스토어/DB에서 유사 판례 검색
    print("🔍 유사 판례 검색 중…")
    results = u.retrieve_db(
        user_query,
        HOST, PORT, USER, PASSWORD, DB_NAME,
        k=k,
        base_db_dir = BASE_DB_DIR
    )

    if not results or len(results) == 0:
        return "유사 판례를 찾지 못했습니다."

    # 가장 유사한 판례 1건만 사용 (확장 가능)
    top = results[0]
    context = f"{top['유사문단']} [ref:{top['판례일련번호']}]"
    full_document = top['전문']

    # 2) LLM 답변 생성 (get_llm 사용)
    print("🖋️ 답변 생성 중…", end=" ")
    start = time.perf_counter()
    prompt = answer_tpl.render(
        context=context,
        full_document=full_document,
        user_query=user_query
    )
    llm = get_llm()
    result = llm.invoke(prompt)
    print(f"✔ ({time.perf_counter()-start:.1f}s)")

    return result.content if hasattr(result, "content") else result

def main():
    user_query = input("💬 질문을 입력하세요: ").strip()
    answer = run_rag(user_query)
    print("\n📌 최종 요약\n", answer)
    return answer

if __name__ == "__main__":
    main()
