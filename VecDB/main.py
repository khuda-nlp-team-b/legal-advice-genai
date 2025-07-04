# main.py
import os
from dotenv import load_dotenv
import utils as u

load_dotenv()

host = os.environ.get('DB_HOST')
port = 3306
username = os.environ.get('DB_USER')
password = os.getenv('DB_PASSWORD')
db = os.getenv('DB_NAME')
api_key = os.getenv('OPENAI_API_KEY')

#cdb.save_df(host,port,username,password,db)
#cdb.create_db()
u.retrieve_db('차를 주차장에 주차해놓고 왔는데, 돌아와보니 차량이 훼손되어 있었다. CCTV를 확인해보니 주차장 관리자가 차량을 이동하다가 발생한 사고였다.',host,port,username,password,db,k=3)