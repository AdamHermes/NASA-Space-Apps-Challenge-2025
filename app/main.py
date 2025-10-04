from fastapi import FastAPI
from app.routes import router
from .routers import data, ml_routers, visualization

# Create FastAPI app
app = FastAPI(
    title="My FastAPI App",
    description="A sample FastAPI project with app folder",
    version="1.0.0"
)
    

app.include_router(router)
app.include_router(data.router)
app.include_router(ml_routers.router)
app.include_router(visualization.router)