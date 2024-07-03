from fastapi import FastAPI
from .routers import root, predict

app = FastAPI()

app.include_router(root.router)
app.include_router(predict.router)
