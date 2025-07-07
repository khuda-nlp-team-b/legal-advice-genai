import os
from dotenv import load_dotenv
import pymysql
from jinja2 import Template
from utils import util as u

# 1. 환경변수 로드
def main():
    load_dotenv()

    # 2. DB 연결 (DictCursor로 수정)
    conn = pymysql.connect(
        host=os.environ['DB_HOST'],
        port=int(os.environ.get('DB_PORT', 3306)),
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        db=os.environ['DB_NAME'],
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor
    )

    # 3. 벡터스토어 로드
    vectorstore = u.setup_db(base_db_dir='./db')

    # 4. 템플릿 로드
    with open('prompts/answer_synth.j2', encoding='utf-8') as f:
        answer_tpl = Template(f.read())

    # 5. 유저 쿼리 입력
    user_query = input("질문을 입력하세요: ")

    # 6. 답변 생성 (RAG)
    openai_key = os.environ['OPENAI_API_KEY']
    answer = u.run_rag(
        user_query=user_query,
        vectorstore=vectorstore,
        k=5,
        conn=conn,
        answer_tpl=answer_tpl,
        openai_key=openai_key
    )
    print("\n===== LLM 답변 =====\n")
    print(answer)

if __name__ == "__main__":
    main() 