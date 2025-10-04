from fastapi import FastAPI
from app.routes import router
from .routers import data, ml_routers, visualization
from fastapi.middleware.cors import CORSMiddleware
# Create FastAPI app
app = FastAPI(
    title="My FastAPI App",
    description="A sample FastAPI project with app folder",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Your frontend URL
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],  # Include OPTIONS
    allow_headers=["*"],
)

    

app.include_router(router)
app.include_router(data.router)
app.include_router(ml_routers.router)
app.include_router(visualization.router)