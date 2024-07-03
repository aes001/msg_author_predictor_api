from app import app
import uvicorn
from dotenv import load_dotenv
import os

load_dotenv()

if __name__ == "__main__":
    try:
        host = os.getenv("HOST", "127.0.0.1")
        port = int(os.getenv("PORT", "8000"))
        reload = os.getenv("RELOAD", "True").lower() == "true"
        log_level = os.getenv("LOG_LEVEL", "info")
        uvicorn.run("main:app", host=host, port=port,
                    reload=reload, log_level=log_level)
    except KeyboardInterrupt:
        pass
