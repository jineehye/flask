# Flask용 Dockerfile
FROM python:3.9-slim

# 작업 디렉토리 설정
WORKDIR /app

# 의존성 파일 복사
COPY requirements.txt requirements.txt

# 의존성 설치
RUN pip install --no-cache-dir -r requirements.txt

# 앱 파일 복사
COPY . .

# Flask 앱 실행
CMD ["python", "demo.py"]