from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
import pandas as pd
import pymysql
from langchain.schema import Document
import os

def load_df(host, port, username, password, db):
    conn = pymysql.connect(host=host, port=port, user=username, password=password, db=db)
    df_판례 = pd.read_sql('SELECT * FROM 판례 where 사건종류명 = \'형사\' limit 300;', conn)
    conn.close()
    return df_판례

case_type_list = [
        '형사', '민사', '일반행정', '세무', '특허', '가사', '가사_상속', '가사_이혼', '가사_재산분할',
        '국가배상', '근로_산재', '근로_임금', '기업_공동행위', '기업_공정위', '기업_해고',
        '제조물_책임_민사', '행정_시정명령', '행정_정보공개', '행정강제', '행정', '형사_교통', '형사_폭행'
    ]

def get_case_type_index(case_type):
    try:
        return case_type_list.index(case_type)
    except ValueError:
        raise ValueError(f"[오류] '{case_type}' 는 유효한 사건 유형이 아닙니다.")

def build_rag_db_by_case_type(df_판례, case_type, api_key, base_db_dir='./db'):
    """
    특정 사건 유형에 해당하는 판례 내용만 사용하여 RAG용 벡터 DB를 생성합니다.

    Args:
        df_판례 (pd.DataFrame): 판례 데이터프레임 (필수 컬럼: '사건종류명', '판례내용')
        case_type (str): 예: '형사', '민사', '일반행정' 등
        api_key (str): OpenAI API 키
        base_db_dir (str): 기본 DB 저장 경로

    Returns:
        db (Chroma): 생성된 Chroma 벡터 DB 객체
    """

    # 1. 해당 사건 유형에 해당하는 행 필터링
    filtered_df = df_판례[df_판례['사건종류명'] == case_type].reset_index(drop=True)
    if filtered_df.empty:
        raise ValueError(f"[경고] '{case_type}' 유형에 해당하는 판례가 없습니다.")

    # 2. Document 객체로 변환
    documents = []

    '''  for i, content in enumerate(filtered_df['판례내용']):
            doc_id = f"{case_type}_{i}"
            doc = Document(page_content=content, metadata={"source": doc_id})
            documents.append(doc)
            original_text_map[doc_id] = content'''
        
    for i, row in filtered_df.iterrows():
        metadata = {
            "source": row['판례일련번호'],
            'case_type': row['사건종류명'],
            'full_content': row['판례내용']
        }
        doc = Document(page_content=row['판례내용'], metadata=metadata)
        documents.append(doc)

    # 3. 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=700,
        chunk_overlap=100,
        encoding_name='cl100k_base'
    )
    split_docs = text_splitter.split_documents(documents)

    # 4. DB 경로 지정 및 벡터 저장
    db_dir = os.path.join(base_db_dir, f"chromadb_law_{case_type}")
    os.makedirs(db_dir, exist_ok=True)

    embeddings_model = OpenAIEmbeddings(api_key=api_key)

    case_num = get_case_type_index(case_type)

    db = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings_model,
        collection_name=f'LAW_RAG_{case_num}',
        persist_directory=db_dir,
        collection_metadata={'hnsw:space': 'cosine'}
    )

    print(f"[완료] '{case_type}' 유형의 판례 {len(documents)}건이 DB에 저장되었습니다.")

def search_law_rag(query, case_type, api_key, k=1):
    """
    특정 사건 유형 DB에서 질의를 검색하여 유사 문단과 전체 판례 원문을 출력합니다.

    Args:
        query (str): 사용자 질문
        case_type (str): 사건 유형 (예: '형사', '민사', ...)
        api_key (str): OpenAI API 키
        text_map (dict): {source_id: 판례 원문} 형태의 dict
        k (int): 검색할 유사 문단 개수 (기본값 1)
    """

    # 사건 유형 번호 → collection_name 생성
    case_num = get_case_type_index(case_type)

    if not os.path.exists(f"./db/chromadb_law_{case_type}"):
        raise ValueError(f"[오류] '{case_type}' 유형의 DB가 존재하지 않습니다. 먼저 DB를 생성하세요.")
    
    # DB 로드
    db = Chroma(
        persist_directory=f"./db/chromadb_law_{case_type}",
        collection_name=f"LAW_RAG_{case_num}",
        embedding_function=OpenAIEmbeddings(api_key=api_key)
    )

    '''    # 유사도 검색
        results = db.similarity_search(query, k=k, filter={"case_type": case_type})'''  
    
    retriever = db.as_retriever(
        search_type = 'similarity', search_kwargs={'k': k}
        )
    results = retriever.invoke(query)
    

    # 결과 출력
    for i, doc in enumerate(results):
        meta = doc.metadata
        print(f"\n🔍 [결과 {i+1}]")
        print(f"▶ 판례일련번호 : {meta['source']}")
        print("▶ 유사 문단:", doc.page_content.strip())

        print("\n📄 [전체 판례 내용]")
        full_content = meta.get('full_content', '전체 판례 내용이 없습니다.')
        print(full_content.strip())
        print("\n" + "="*50)
    
