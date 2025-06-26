import pandas as pd
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils import create_db, retrieve_db

# 데이터 불러오기
api_key = "MY_KEY"
data = pd.read_csv('판례_데이터_가공.csv', encoding='utf-8-sig')

# 벡터 DB 생성
if not os.path.exists('db'):
    create_db(data, api_key)

# 벡터 DB 검색
query = '누가 나를 칼로 찔렀어' #'누가 나를 주먹으로 계속 때렸어' # '누가 나를 칼로 찔렀어'
retrieve_db(query, api_key, data)