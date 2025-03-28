from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .endpoint import student_performance

app = FastAPI()

origins = [
    "http://localhost:5500",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include the student performance prediction endpoint
app.include_router(student_performance.router)
