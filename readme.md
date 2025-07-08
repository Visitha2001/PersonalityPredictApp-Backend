python -m venv venv
python --version
.\venv\Scripts\activate
cd app
uvicorn main:app --reload --port 8008