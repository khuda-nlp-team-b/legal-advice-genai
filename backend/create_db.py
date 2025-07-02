from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
import pandas as pd
import pymysql
from langchain_huggingface import HuggingFaceEmbeddings
import os
import warnings
import chromadb
from dotenv import load_dotenv

warnings.filterwarnings('ignore')

# Chroma telemetry 비활성화
os.environ["CHROMA_TELEMETRY_ENABLED"] = "false"

# 환경변수 로드
load_dotenv()

# DB 설정
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = int(os.getenv("DB_PORT", 3306))
DB_USER = os.getenv("DB_USER", "root")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")
DB_NAME = os.getenv("DB_NAME", "lawdb")

def get_mysql_connection(host, port, username, password, db_name):
    try:
        conn = pymysql.connect(
            host=host, port=port, user=username, password=password,
            db=db_name, charset='utf8',
            cursorclass=pymysql.cursors.DictCursor
        )
        print("MySQL 연결 성공")
        return conn
    except Exception as e:
        print(f"MySQL 연결 오류: {e}")
        return None

def save_df(host, port, username, password, db_name, csv_path):
    try:
        conn = pymysql.connect(host=host, port=port, user=username, password=password, db=db_name)
        print("판례 데이터를 MySQL에서 200건만 불러오는 중")
        df = pd.read_sql('SELECT * FROM 판례 LIMIT 200', conn)
        df.to_csv(csv_path, index=False)
        conn.close()
        print(f"CSV로 저장 완료: {csv_path}")
    except Exception as e:
        print(f"CSV 저장 오류: {e}")

def create_db(base_db_dir='./db'):
    print("CPU 기반 벡터 DB 생성 시작")

    csv_path = '판례.csv'
    save_df(DB_HOST, DB_PORT, DB_USER, DB_PASSWORD, DB_NAME, csv_path)

    if not os.path.exists(csv_path):
        print("판례.csv 파일이 존재하지 않습니다.")
        return

    try:
        df_판례 = pd.read_csv(csv_path)
    except Exception as e:
        print(f"CSV 로드 오류: {e}")
        return

    if df_판례.empty:
        print("판례.csv 파일이 비어 있습니다.")
        return

    try:
        client = chromadb.PersistentClient(path=base_db_dir)
        client.delete_collection(name='LAW_RAG')
        print(f"기존 벡터 DB 삭제 완료: {base_db_dir}")
    except Exception as e:
        print(f"기존 DB 삭제 중 오류: {e}")

    os.makedirs(base_db_dir, exist_ok=True)

    print('Document 객체 리스트 생성 중')
    docs = []
    for _, row in df_판례.iterrows():
        content = str(row['판례내용'])
        if len(content) > 100000:
            continue
        metadata = {
            "source": row['판례일련번호'],
            "case_type": row.get('사건명', '알 수 없음')
        }
        docs.append(Document(page_content=content, metadata=metadata))
    print(f'Document 생성 완료: {len(docs)}건')

    print('HuggingFaceEmbeddings 로드 중 (CPU)')
    embeddings = HuggingFaceEmbeddings(
        model_name="jhgan/ko-sbert-nli",
        model_kwargs={"device": "cpu"}
    )
    print('HuggingFaceEmbeddings 로드 완료')

    print('텍스트 분할 중')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=150
    )
    split_docs = text_splitter.split_documents(docs)
    print(f'텍스트 분할 완료: {len(split_docs)} 청크 생성')

    print('벡터스토어 생성 및 적재 중')
    total_docs = len(split_docs)
    batch_size = 5000
    processed = 0

    for i in range(0, total_docs, batch_size):
        batch_docs = split_docs[i:i + batch_size]
        processed += len(batch_docs)

        if i == 0:
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                collection_name='LAW_RAG',
                persist_directory=base_db_dir
            )
        else:
            vectorstore.add_documents(batch_docs)

        print(f'진행률: {processed}/{total_docs} ({processed / total_docs * 100:.1f}%)')

    vectorstore.persist()
    print('벡터스토어 생성 및 저장 완료')
    print(f"{len(docs)}건 판례가 원본과 청크로 분리되어 {base_db_dir}/LAW_RAG 에 저장 완료")

if __name__ == "__main__":
    try:
        create_db(base_db_dir='./db')
    except Exception as e:
        print(f"create_db 실행 중 오류: {e}")
