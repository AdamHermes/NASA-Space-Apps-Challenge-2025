from fastapi import FastAPI
from app.routes import router
from .routers import data

# Create FastAPI app
app = FastAPI(
    title="My FastAPI App",
    description="A sample FastAPI project with app folder",
    version="1.0.0"
)
    
# Include routes from routes.py
app.include_router(router)
app.include_router(data.router)