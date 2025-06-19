from pydantic_settings import BaseSettings
import os

class Settings(BaseSettings):
    GROK_API_KEY: str
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), '..', 'data')
    SINGULAR_CSV: str = os.path.join(DATA_DIR, 'singular.csv')
    COMBINED_CSV: str = os.path.join(DATA_DIR, 'all_combined_ornaments.csv')

    class Config:
        env_file = ".env"

settings = Settings()
