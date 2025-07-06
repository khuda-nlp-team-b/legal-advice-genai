# main.py
import os
from dotenv import load_dotenv
import util as u
import pymysql
import pandas as pd
import torch
import time
from datetime import datetime, timedelta
from langchain_community.vectorstores import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain.embeddings import SentenceTransformerEmbeddings

load_dotenv()


def save_df(host,port,username,password,db_name):
    conn = pymysql.connect(host=host, port=port, user=username, password=password, db=db_name)
    df = pd.read_sql('SELECT * FROM 판례', conn)
    df.to_csv('판례.csv', index=False)
    conn.close()

def create_db_from_mysql(host, port, username, password, db_name, base_db_dir='./db'):
    """MySQL에서 직접 데이터를 읽어서 벡터 DB를 구축"""
    print("🔄 MySQL에서 판례 데이터를 읽어오는 중...")
    
    # MySQL 연결 및 데이터 읽기
    conn = pymysql.connect(host=host, port=port, user=username, password=password, db=db_name)
    df_판례 = pd.read_sql('SELECT * FROM 판례', conn)
    conn.close()
    
    print(f"📊 총 {len(df_판례)}개의 판례 데이터를 로드했습니다")
    
    # CUDA 사용 가능 여부 확인
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("❌ CUDA 사용 불가능 - CPU 모드로 실행됩니다")
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 기존 벡터 DB 디렉토리가 존재하면 완전 삭제
    if os.path.exists(base_db_dir):
        try:
            import shutil
            shutil.rmtree(base_db_dir)
            print(f"[삭제] 기존 벡터 DB 디렉토리가 완전히 삭제되었습니다: {base_db_dir}")
        except Exception as e:
            print(f"[경고] 기존 DB 디렉토리 삭제 중 오류 발생: {e}")
    
    # DB 디렉토리 생성
    os.makedirs(base_db_dir, exist_ok=True)
    
    # Document 객체 리스트 생성
    print('Document 객체 리스트 생성 중...')
    docs = []
    for i, row in df_판례.iterrows():
        metadata = {
            "source": row['판례일련번호'],
            'case_type': row['사건명']
        }
        doc = Document(page_content=str(row['판례내용']), metadata=metadata)
        docs.append(doc)
    print('Document 객체 리스트 생성 완료')
    
    # 임베딩 모델 설정 (CUDA 사용 여부에 따라)
    device = "cuda" if cuda_available else "cpu"
    embeddings = SentenceTransformerEmbeddings(model_name='nlpai-lab/KURE-v1', model_kwargs={"device": device})
    print(f"🤖 임베딩 모델 로드 완료 (장치: {device})")
    
    # 벡터스토어 생성 (chunk_size=500, chunk_overlap=75)
    print('\n=== 벡터스토어 생성 (chunk_size=500, chunk_overlap=75) ===')
    chunk_start_time = time.time()
    
    print('텍스트 분할 중...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75
    )
    
    split_docs = text_splitter.split_documents(docs)
    print(f'텍스트 분할 완료: {len(docs)}개 문서 → {len(split_docs)}개 청크')
    
    # 벡터스토어 생성
    print('벡터스토어 생성 중...')
    total_docs = len(split_docs)
    print(f'총 {len(docs)}개의 문서를 {total_docs}개의 청크로 처리합니다...')
    
    # 배치 크기 설정 (메모리 관리를 위해)
    batch_size = 1000
    processed = 0
    processing_times = []  # 각 배치의 처리 시간을 저장할 리스트
    
    for i in range(0, total_docs, batch_size):
        batch_start_time = time.time()
        batch_docs = split_docs[i:i + batch_size]
        processed += len(batch_docs)
        
        if i == 0:
            # 첫 번째 배치로 벡터스토어 생성
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                collection_name='LAW_RAG_500_75',
                persist_directory=base_db_dir
            )
        else:
            # 나머지 배치는 추가
            vectorstore.add_documents(batch_docs)
        
        # 배치 처리 시간 계산
        batch_time = time.time() - batch_start_time
        processing_times.append(batch_time)
        
        # 이전 배치들의 평균 처리 시간으로 예상 완료시간 계산
        if len(processing_times) > 1:
            avg_time_per_batch = sum(processing_times) / len(processing_times)
            remaining_batches = (total_docs - processed) // batch_size + (1 if (total_docs - processed) % batch_size > 0 else 0)
            estimated_remaining_time = remaining_batches * avg_time_per_batch
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            print(f'진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%) - 예상 완료: {estimated_completion.strftime("%H:%M:%S")} (평균 배치시간: {avg_time_per_batch:.1f}초)')
        else:
            # 첫 번째 배치 후에는 아직 평균을 계산할 수 없으므로 현재 배치 시간으로 추정
            remaining_batches = (total_docs - processed) // batch_size + (1 if (total_docs - processed) % batch_size > 0 else 0)
            estimated_remaining_time = remaining_batches * batch_time
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            print(f'진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%) - 예상 완료: {estimated_completion.strftime("%H:%M:%S")} (현재 배치시간: {batch_time:.1f}초)')
    
    vectorstore.persist()
    chunk_time = time.time() - chunk_start_time
    print(f'✅ 벡터스토어 생성 완료! (소요시간: {chunk_time/60:.1f}분)')
    
    total_time = time.time() - start_time
    print(f'🎉 전체 작업 완료! (총 소요시간: {total_time/60:.1f}분)')

def create_db(base_db_dir='C:\\db'):
    # CUDA 사용 가능 여부 확인
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA 사용 가능: {torch.cuda.get_device_name(0)}")
        print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f}GB")
    else:
        print("❌ CUDA 사용 불가능 - CPU 모드로 실행됩니다")
    
    # 시작 시간 기록
    start_time = time.time()
    
    # 현재 스크립트 디렉토리 기준으로 파일 경로 설정
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(script_dir, '판례.csv')
    
    # 파일 존재 여부 확인
    if not os.path.exists(csv_file_path):
        print(f"❌ 파일을 찾을 수 없습니다: {csv_file_path}")
        print(f"현재 디렉토리: {os.getcwd()}")
        print(f"스크립트 디렉토리: {script_dir}")
        print("사용 가능한 파일들:")
        for file in os.listdir(script_dir):
            if file.endswith('.csv'):
                print(f"  - {file}")
        return
    
    # 판례.csv 파일 읽기
    print(f"📁 파일 경로: {csv_file_path}")
    df_판례 = pd.read_csv(csv_file_path)
    print(f"📊 총 {len(df_판례)}개의 판례 데이터를 로드했습니다")

    # 기존 벡터 DB 디렉토리가 존재하면 완전 삭제
    if os.path.exists(base_db_dir):
        try:
            import shutil
            shutil.rmtree(base_db_dir)
            print(f"[삭제] 기존 벡터 DB 디렉토리가 완전히 삭제되었습니다: {base_db_dir}")
        except Exception as e:
            print(f"[경고] 기존 DB 디렉토리 삭제 중 오류 발생: {e}")
    
    # DB 디렉토리 생성
    os.makedirs(base_db_dir, exist_ok=True)
    
    # Document 객체 리스트 생성
    print('Document 객체 리스트 생성 중...')
    docs = []
    for i, row in df_판례.iterrows():
        metadata = {
            "source": row['판례일련번호'],
            'case_type': row['사건명']
        }
        doc = Document(page_content=str(row['판례내용']), metadata=metadata)
        docs.append(doc)
    print('Document 객체 리스트 생성 완료')
    # model_name = 'intfloat/multilingual-e5-large-instruct'
    embeddings = SentenceTransformerEmbeddings(model_name='nlpai-lab/KURE-v1', model_kwargs={"device": "cpu"})
    
    print('텍스트 분할 중...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250, ##### 1500
        chunk_overlap=25 ##### 300
    )
    split_docs = text_splitter.split_documents(docs)
    print(f'텍스트 분할 완료: {len(docs)}개 문서 → {len(split_docs)}개 청크')
    
    # 임베딩 모델 설정 (CUDA 사용 여부에 따라)
    device = "cuda" if cuda_available else "cpu"
    embeddings = SentenceTransformerEmbeddings(model_name='nlpai-lab/KURE-v1', model_kwargs={"device": device})
    print(f"🤖 임베딩 모델 로드 완료 (장치: {device})")
    
    # 첫 번째 벡터스토어 생성 (chunk_size=250, chunk_overlap=50)
    print('\n=== 첫 번째 벡터스토어 생성 (chunk_size=250, chunk_overlap=50) ===')
    chunk_start_time = time.time()
    
    print('텍스트 분할 중...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=250,
        chunk_overlap=50
    )
    
    split_docs = text_splitter.split_documents(docs)
    print(f'텍스트 분할 완료: {len(docs)}개 문서 → {len(split_docs)}개 청크')
    
    # 자식 청크를 저장할 벡터스토어 생성
    print('벡터스토어 생성 중...')
    total_docs = len(split_docs)
    print(f'총 {len(docs)}개의 문서를 {total_docs}개의 청크로 처리합니다...')
    
    # 배치 크기 설정 (메모리 관리를 위해)
    batch_size = 1000
    processed = 0
    processing_times = []  # 각 배치의 처리 시간을 저장할 리스트
    
    for i in range(0, total_docs, batch_size):
        batch_start_time = time.time()
        batch_docs = split_docs[i:i + batch_size]
        processed += len(batch_docs)
        
        if i == 0:
            # 첫 번째 배치로 벡터스토어 생성
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                collection_name='LAW_RAG_250_50',
                persist_directory=base_db_dir
            )
        else:
            # 나머지 배치는 추가
            vectorstore.add_documents(batch_docs)
        
        # 배치 처리 시간 계산
        batch_time = time.time() - batch_start_time
        processing_times.append(batch_time)
        
        # 이전 배치들의 평균 처리 시간으로 예상 완료시간 계산
        if len(processing_times) > 1:
            avg_time_per_batch = sum(processing_times) / len(processing_times)
            remaining_batches = (total_docs - processed) // batch_size + (1 if (total_docs - processed) % batch_size > 0 else 0)
            estimated_remaining_time = remaining_batches * avg_time_per_batch
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            print(f'진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%) - 예상 완료: {estimated_completion.strftime("%H:%M:%S")} (평균 배치시간: {avg_time_per_batch:.1f}초)')
        else:
            # 첫 번째 배치 후에는 아직 평균을 계산할 수 없으므로 현재 배치 시간으로 추정
            remaining_batches = (total_docs - processed) // batch_size + (1 if (total_docs - processed) % batch_size > 0 else 0)
            estimated_remaining_time = remaining_batches * batch_time
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            print(f'진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%) - 예상 완료: {estimated_completion.strftime("%H:%M:%S")} (현재 배치시간: {batch_time:.1f}초)')
    
    vectorstore.persist()
    chunk_time = time.time() - chunk_start_time
    print(f'✅ 첫 번째 벡터스토어 생성 완료! (소요시간: {chunk_time/60:.1f}분)')

    # 두 번째 벡터스토어 생성 (chunk_size=500, chunk_overlap=75)
    print('\n=== 두 번째 벡터스토어 생성 (chunk_size=500, chunk_overlap=75) ===')
    chunk_start_time = time.time()
    
    print('두번째 텍스트 분할 중...')
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=75
    )
    
    split_docs = text_splitter.split_documents(docs)
    print(f'텍스트 분할 완료: {len(docs)}개 문서 → {len(split_docs)}개 청크')
    
    # 자식 청크를 저장할 벡터스토어 생성
    print('벡터스토어 생성 중...')
    total_docs = len(split_docs)
    print(f'총 {len(docs)}개의 문서를 {total_docs}개의 청크로 처리합니다...')
    
    # 배치 크기 설정
    batch_size = 1000
    processed = 0
    processing_times = []  # 각 배치의 처리 시간을 저장할 리스트
    
    for i in range(0, total_docs, batch_size):
        batch_start_time = time.time()
        batch_docs = split_docs[i:i + batch_size]
        processed += len(batch_docs)
        
        if i == 0:
            # 첫 번째 배치로 벡터스토어 생성
            vectorstore = Chroma.from_documents(
                documents=batch_docs,
                embedding=embeddings,
                collection_name='LAW_RAG_500_75',
                persist_directory=base_db_dir
            )
            return
        else:
            # 나머지 배치는 추가
            vectorstore.add_documents(batch_docs)
        
        # 배치 처리 시간 계산
        batch_time = time.time() - batch_start_time
        processing_times.append(batch_time)
        
        # 이전 배치들의 평균 처리 시간으로 예상 완료시간 계산
        if len(processing_times) > 1:
            avg_time_per_batch = sum(processing_times) / len(processing_times)
            remaining_batches = (total_docs - processed) // batch_size + (1 if (total_docs - processed) % batch_size > 0 else 0)
            estimated_remaining_time = remaining_batches * avg_time_per_batch
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            print(f'진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%) - 예상 완료: {estimated_completion.strftime("%H:%M:%S")} (평균 배치시간: {avg_time_per_batch:.1f}초)')
        else:
            # 첫 번째 배치 후에는 아직 평균을 계산할 수 없으므로 현재 배치 시간으로 추정
            remaining_batches = (total_docs - processed) // batch_size + (1 if (total_docs - processed) % batch_size > 0 else 0)
            estimated_remaining_time = remaining_batches * batch_time
            estimated_completion = datetime.now() + timedelta(seconds=estimated_remaining_time)
            
            print(f'진행률: {processed}/{total_docs} ({processed/total_docs*100:.1f}%) - 예상 완료: {estimated_completion.strftime("%H:%M:%S")} (현재 배치시간: {batch_time:.1f}초)')
    
    vectorstore.persist()
    chunk_time = time.time() - chunk_start_time
    print(f'✅ 두 번째 벡터스토어 생성 완료! (소요시간: {chunk_time/60:.1f}분)')
    
    # 전체 완료 시간 계산
    total_time = time.time() - start_time
    print(f"\n🎉 전체 작업 완료!")
    print(f"📊 총 소요시간: {total_time/60:.1f}분")
    print(f"📁 저장 위치: {base_db_dir}")
    print(f"[완료] 판례 {len(docs)}건이 원본과 청크로 분리되어 저장되었습니다.")

host = os.environ.get('DB_HOST')
port = 3306
username = os.environ.get('DB_USER')
password = os.getenv('DB_PASSWORD')
db = os.getenv('DB_NAME')
api_key = os.getenv('OPENAI_API_KEY')

#cdb.save_df(host,port,username,password,db)
#cdb.create_db()
#u.retrieve_db('택시를 타고 가다가, 운전자가 신호를 위반하며 다른 차량과 충돌하는 사고가 발생했다. 택시 회사는 사고에 대한 책임을 인정하지 않고 있다.',host,port,username,password,db,k=3)

if __name__ == "__main__":
    # MySQL에서 직접 벡터 DB 구축 (권장)
    create_db_from_mysql(
        host='woodjudge-db.cp0w6o200jli.ap-northeast-2.rds.amazonaws.com',
        port=3306,
        username='admin',
        password='woodjudgedbpw',
        db_name='woodjudgeDB',
        base_db_dir='./db'
    )