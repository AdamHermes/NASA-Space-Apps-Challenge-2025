from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routes import router
from .routers import data, ml_routers, visualization,train_routers,light_curve
from fastapi.staticfiles import StaticFiles
from pathlib import Path

# Create FastAPI app
app = FastAPI(
    title="My FastAPI App",
    description="A sample FastAPI project with app folder",
    version="1.0.0"
)
    
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "https://nasa-space-apps-challenge-2025-frontend1.onrender.com"

        # add your deployed frontend origins here
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


app.include_router(router)
app.include_router(data.router)
app.include_router(ml_routers.router)
app.include_router(visualization.router)
app.include_router(train_routers.router)
app.include_router(light_curve.router)
# app.include_router(merge_csvs.router)
