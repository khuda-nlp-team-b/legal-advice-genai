from fastapi import FastAPI
from pydantic import BaseModel
from utils import util as u
import os
from dotenv import load_dotenv
import pymysql
from jinja2 import Template
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
load_dotenv()

# CORS 허용 (프론트와 포트가 다를 경우 필요)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class AskRequest(BaseModel):
    question: str

@app.post("/api/ask")
def ask(req: AskRequest):
    conn = pymysql.connect(
        host=os.environ['DB_HOST'],
        port=int(os.environ.get('DB_PORT', 3306)),
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        db=os.environ['DB_NAME'],
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor
    )
    vectorstore = u.setup_db(base_db_dir='./db')
    with open('prompts/answer_synth.j2', encoding='utf-8') as f:
        answer_tpl = Template(f.read())
    answer = u.run_rag(
        user_query=req.question,
        vectorstore=vectorstore,
        k=5,
        conn=conn,
        answer_tpl=answer_tpl,
        openai_key=os.environ['OPENAI_API_KEY']
    )
    return {"answer": answer}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 