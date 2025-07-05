import os
import argparse
import jinja2
from dotenv import load_dotenv
from utils import util as u

load_dotenv()

# ────────────────── 1) 경로·환경 ──────────────────
BASE_PATH  = os.path.dirname(os.path.abspath(__file__))
PROMPT_DIR = os.path.join(BASE_PATH, "prompts")
DB_SUBDIR = "LAW_RAG_500_75"

OPENAI_KEY = os.getenv("OPENAI_API_KEY")

# ────────────────── 2) DB 접속 정보 ──────────────────
HOST        = os.getenv("DB_HOST")
PORT        = int(os.getenv("DB_PORT", 3306))
USER        = os.getenv("DB_USER")
PASSWORD    = os.getenv("DB_PASSWORD")
DB_NAME     = os.getenv("DB_NAME")

# ────────────────── 3) Jinja2 템플릿 로드 ──────────────────
env = jinja2.Environment(
    loader=jinja2.FileSystemLoader(PROMPT_DIR),
    autoescape=False,
    trim_blocks=True,
    lstrip_blocks=True,
)

answer_tpl = env.get_template("answer_synth.j2")

def main():
    vectorstore = u.setup_db()
    conn = u.get_mysql_connection(HOST,PORT,USER,PASSWORD,DB_NAME)
    user_query = input("💬 처한 법적 상황과 걱정하는 점을 알려주세요: ").strip()
    answer = u.run_rag(user_query,vectorstore,5,conn,answer_tpl,OPENAI_KEY)
    print(user_query)
    print("📌 최종 요약\n", answer)
    
    '''while True:
        user_query = input("해당 내용에 대해 추가적으로 궁금한 점이 있으신가요? ").strip()
        if user_query == 'exit':
            break
        answer = u.run_rag(user_query,vectorstore,5,HOST,PORT,USER,PASSWORD,DB_NAME)
        print(user_query)
        print("📌 답변 \n", answer)'''
    
    #return answer

if __name__ == "__main__":
    main()