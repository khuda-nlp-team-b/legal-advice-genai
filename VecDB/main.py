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
cdb.create_db()
#cdb.retrieve_db('누가 내 돈을 가지고 다른 곳에 투자하여서 모두 소진했어.',host,port,username,password,db,api_key)