# LLaVA 리포트 프론트엔드 + API 로컬 실행 가이드

## 필수 설치

1. **Node.js** (https://nodejs.org)
2. **Python 3.10+** (https://python.org)
3. **PostgreSQL** (https://www.postgresql.org/download/)
   - 설치 시 비밀번호: `1234`, 포트: `5432` 로 설정

## 설정 순서

### 1. Python 패키지 설치
```bash
pip install psycopg2-binary fastapi uvicorn
```

### 2. PostgreSQL DB 생성 (1회)
```bash
psql -U postgres -c "CREATE DATABASE llava_reports;"
```
비밀번호 입력: `1234`

### 3. 데이터 import (1회)
```bash
python scripts/import_llava_reports.py --json data/llava_reports_backup.json
```

### 4. API 서버 실행
```bash
python _test_server.py
```
http://localhost:8000/llava/reports 에서 JSON 확인

### 5. 프론트엔드 실행
```bash
cd Frontend
npm install
npm run dev
```
http://localhost:5173 에서 확인

## 폴더 구조
```
_test_server.py                    # API 서버 (포트 8000)
apps/api/llava.py                  # API 엔드포인트
src/storage/llava_db.py            # PostgreSQL DB 모듈
data/llava_reports_backup.json     # 샘플 데이터 10건
scripts/import_llava_reports.py    # 데이터 import 스크립트
Frontend/                          # React 프론트엔드
```
