import test_utils as tu
from dotenv import load_dotenv
import os

load_dotenv()
api_key = os.getenv('OPENAI_API_KEY')

tu.create_test_db(api_key=api_key,base_db_dir='./db_test')
