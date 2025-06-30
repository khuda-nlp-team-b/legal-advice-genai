import test_utils as tu
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

tu.create_test_db(api_key=api_key,base_db_dir='./db_test',chunk_size=250,chunk_overlap=25)
tu.create_test_db(api_key=api_key,base_db_dir='./db_test',chunk_size=500,chunk_overlap=50)
tu.create_test_db(api_key=api_key,base_db_dir='./db_test',chunk_size=1000,chunk_overlap=100)
