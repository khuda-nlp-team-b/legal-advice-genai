# main.py
import os
from dotenv import load_dotenv
import create_db as cdb

load_dotenv()

host = os.environ.get('DB_HOST')
port = 3306
username = os.environ.get('DB_USER')
password = os.getenv('DB_PASSWORD')
db = os.getenv('DB_NAME')
api_key = os.getenv('OPENAI_API_KEY')

#cdb.save_df(host,port,username,password,db)
#cdb.create_db()
cdb.retrieve_db('상점을 운영하는데, 갑작스럽게 임대료를 인상하겠다는 건물주의 통보를 받았다. 계약서에는 임대료 인상에 대한 내용이 명확히 명시되어 있지 않은 상황이다.',host,port,username,password,db,k=3)