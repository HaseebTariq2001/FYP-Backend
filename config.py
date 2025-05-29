import os
from dotenv import load_dotenv

load_dotenv()  # Load from .env file

class Config:
    MYSQL_HOST = os.getenv('MYSQL_HOST')
    MYSQL_USER = os.getenv('MYSQL_USER')
    MYSQL_PASSWORD = os.getenv('MYSQL_PASSWORD')
    MYSQL_DB = os.getenv('MYSQL_DB')

    MAX_CONTENT_LENGTH = int(os.getenv('MAX_CONTENT_LENGTH', 50 * 1024 * 1024))  # Optional default
    UPLOAD_FOLDER = os.getenv('UPLOAD_FOLDER', 'uploads')
    SECRET_KEY = os.getenv('SECRET_KEY', 'fallback_secret_key')  # Use fallback in dev
