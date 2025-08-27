#!/usr/bin/env python3
"""
Simple test API to diagnose the startup issue
"""

from fastapi import FastAPI
import os

app = FastAPI(title="Simple Test API")

@app.get("/")
async def root():
    return {"message": "Simple API is working!", "profit_key": bool(os.getenv("PROFIT_API_KEY"))}

@app.get("/health")
async def health():
    return {"status": "healthy", "test": "simple"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
