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
cdb.retrieve_db('나는 친구에게 500만 원을 빌려줬는데, 약속한 기한이 지나도 갚지 않고 연락을 피하고 있다. 여러 차례 연락했지만 계속 무시당하고 있다. 법적으로 대응해야 할지 고민 중이다.',host,port,username,password,db,k=3)