#!/usr/bin/env python3
"""
Railway Migration Endpoint
Basit FastAPI endpoint - migration trigger için
"""
from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import subprocess
import os

app = FastAPI(title="Migration API", version="1.0.0")

@app.post("/migrate")
async def run_migration():
    """Run CSV to PostgreSQL migration"""
    try:
        # Check CSV files
        csv_parts = []
        for suffix in ['aa', 'ab', 'ac', 'ad']:
            gz_file = f"enhanced_stock_data_part_{suffix}.gz"
            if os.path.exists(gz_file):
                csv_parts.append(gz_file)
        
        if not csv_parts:
            return JSONResponse(
                status_code=404,
                content={"error": "CSV files not found", "files_searched": ["enhanced_stock_data_part_*.gz"]}
            )
        
        # Run migration
        print("🚀 Starting migration via endpoint...")
        result = subprocess.run([
            "python", "csv_to_postgresql.py"
        ], capture_output=True, text=True, timeout=600)  # 10 min timeout
        
        if result.returncode == 0:
            return {
                "success": True,
                "message": "🎉 Migration completed!",
                "csv_files": csv_parts,
                "stdout": result.stdout[-2000:],  # Last 2000 chars
                "stderr": result.stderr[-1000:] if result.stderr else ""
            }
        else:
            return JSONResponse(
                status_code=500,
                content={
                    "error": "❌ Migration failed",
                    "returncode": result.returncode,
                    "stdout": result.stdout[-2000:],
                    "stderr": result.stderr[-1000:] if result.stderr else ""
                }
            )
    except subprocess.TimeoutExpired:
        return JSONResponse(
            status_code=408,
            content={"error": "⏰ Migration timeout after 10 minutes"}
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": f"💥 Exception: {str(e)}"}
        )

@app.get("/")
async def root():
    return {"message": "Migration API ready", "endpoint": "/migrate"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
