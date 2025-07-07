import os
import pymysql
from dotenv import load_dotenv

load_dotenv()

def test_case_lookup():
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
            # 전체 판례 개수 확인
            cursor.execute("SELECT COUNT(*) as total FROM 판례")
            total = cursor.fetchone()['total']
            print(f"전체 판례 개수: {total}")
            
            # 판례번호 2618 확인
            cursor.execute("SELECT 판례일련번호, LEFT(판례내용, 200) as preview FROM 판례 WHERE 판례일련번호 = 2618")
            result = cursor.fetchone()
            
            if result:
                print(f"판례번호 2618 찾음!")
                print(f"미리보기: {result['preview']}")
            else:
                print("판례번호 2618을 찾을 수 없습니다.")
                
                # 비슷한 번호들 확인
                cursor.execute("SELECT 판례일련번호 FROM 판례 WHERE 판례일련번호 BETWEEN 2600 AND 2700 ORDER BY 판례일련번호 LIMIT 10")
                similar = cursor.fetchall()
                print(f"2600-2700 범위의 판례번호들: {[r['판례일련번호'] for r in similar]}")
                
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        conn.close()

if __name__ == "__main__":
    test_case_lookup() 
