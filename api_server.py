from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from utils import util as u
import os
from dotenv import load_dotenv
import pymysql
from jinja2 import Template
from fastapi.middleware.cors import CORSMiddleware
import json

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

# 전역 변수로 벡터스토어와 템플릿 저장 (한 번만 초기화)
_vectorstore = None
_answer_template = None

def get_vectorstore():
    global _vectorstore
    if _vectorstore is None:
        print("🔄 벡터스토어 초기화 중...")
        _vectorstore = u.setup_db(base_db_dir='./db')
        print("✅ 벡터스토어 초기화 완료")
    return _vectorstore

def get_answer_template():
    global _answer_template
    if _answer_template is None:
        print("📝 답변 템플릿 로드 중...")
        with open('prompts/answer_synth.j2', encoding='utf-8') as f:
            _answer_template = Template(f.read())
        print("✅ 답변 템플릿 로드 완료")
    return _answer_template

class AskRequest(BaseModel):
    question: str

class CaseRequest(BaseModel):
    caseNumber: str

@app.post("/api/ask")
async def ask(req: AskRequest):
    conn = pymysql.connect(
        host=os.environ['DB_HOST'],
        port=int(os.environ.get('DB_PORT', 3306)),
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        db=os.environ['DB_NAME'],
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor
    )
    vectorstore = get_vectorstore()  # 한 번만 초기화된 벡터스토어 사용
    answer_tpl = get_answer_template()  # 한 번만 로드된 템플릿 사용
    
    async def generate_stream():
        try:
            async for chunk in u.run_rag_stream(
                user_query=req.question,
                vectorstore=vectorstore,
                k=5,
                conn=conn,
                answer_tpl=answer_tpl,
                openai_key=os.environ['OPENAI_API_KEY']
            ):
                yield f"data: {json.dumps({'chunk': chunk})}\n\n"
        except Exception as e:
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        finally:
            conn.close()
    
    return StreamingResponse(
        generate_stream(),
        media_type="text/plain",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "Content-Type": "text/event-stream",
        }
    )

@app.post("/api/case")
async def get_case_info(req: CaseRequest):
    conn = pymysql.connect(
        host=os.environ['DB_HOST'],
        port=int(os.environ.get('DB_PORT', 3306)),
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        db=os.environ['DB_NAME'],
        charset='utf8',
        cursorclass=pymysql.cursors.DictCursor
    )
    
    try:
        with conn.cursor() as cursor:
            sql = f'SELECT 판례일련번호, 판례내용 FROM 판례 WHERE 판례일련번호 = {req.caseNumber}'
            cursor.execute(sql)
            result = cursor.fetchone()
            
            if result:
                return {"caseInfo": result['판례내용']}
            else:
                return {"error": "판례를 찾을 수 없습니다."}, 404
    except Exception as e:
        return {"error": f"데이터베이스 오류: {str(e)}"}, 500
    finally:
        conn.close()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 