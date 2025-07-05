# main.py
import os
from dotenv import load_dotenv
import util as u

load_dotenv()

host = os.environ.get('DB_HOST')
port = 3306
username = os.environ.get('DB_USER')
password = os.getenv('DB_PASSWORD')
db = os.getenv('DB_NAME')
api_key = os.getenv('OPENAI_API_KEY')

#cdb.save_df(host,port,username,password,db)
#cdb.create_db()
u.retrieve_db('택시를 타고 가다가, 운전자가 신호를 위반하며 다른 차량과 충돌하는 사고가 발생했다. 택시 회사는 사고에 대한 책임을 인정하지 않고 있다.',host,port,username,password,db,k=3)