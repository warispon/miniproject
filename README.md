python -m venv venv

pip install -r requirements.txt

python train.py

uvicorn backend.app:app --reload
