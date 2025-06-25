# main.py
import os
from dotenv import load_dotenv
import utils as u

# .env 파일 로드 (가장 먼저 수행)
load_dotenv()

host = os.environ.get('DB_HOST')
port = 3306
username = os.environ.get('DB_USER')
password = os.getenv('DB_PASSWORD')
db = os.getenv('DB_NAME')
api_key = os.getenv('OPENAI_API_KEY')


# 데이터베이스에서 판례 데이터프레임 로드
df_판례 = u.load_df(host, port, username, password, db)
print(df_판례.columns)

# RAG용 벡터 DB 생성
case_type = '형사'  # 예시로 '형사' 사건 유형을 선택
u.build_rag_db_by_case_type(df_판례, case_type, api_key)

# 쿼리 예시
u.search_law_rag(query='지훈이가 나를 때렸어',case_type=case_type, api_key=api_key)