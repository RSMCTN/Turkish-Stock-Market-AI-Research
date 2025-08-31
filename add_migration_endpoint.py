#!/usr/bin/env python3
"""
Main API'ye migration endpoint ekle
"""

# Read main_railway.py
with open('src/api/main_railway.py', 'r') as f:
    content = f.read()

# Migration endpoint to add
migration_endpoint = '''

@app.post("/migrate")
async def run_migration():
    """🚀 Run CSV to PostgreSQL migration on Railway"""
    import subprocess
    import os
    import asyncio
    from pathlib import Path
    
    try:
        logger = logging.getLogger("api.migration")
        logger.info("🚀 Migration başlıyor...")
        
        # Check CSV files
        csv_parts = []
        for suffix in ['aa', 'ab', 'ac', 'ad']:
            gz_file = f"enhanced_stock_data_part_{suffix}.gz"
            if Path(gz_file).exists():
                csv_parts.append(gz_file)
        
        if not csv_parts:
            logger.error("❌ CSV files not found")
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "CSV files not found", 
                    "searched": ["enhanced_stock_data_part_*.gz"],
                    "cwd": str(Path.cwd()),
                    "files": list(Path(".").glob("*.gz"))
                }
            )
        
        logger.info(f"✅ Found {len(csv_parts)} CSV parts: {csv_parts}")
        
        # Run migration in background
        process = await asyncio.create_subprocess_exec(
            "python", "csv_to_postgresql.py",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await asyncio.wait_for(
            process.communicate(), 
            timeout=600  # 10 minutes
        )
        
        if process.returncode == 0:
            logger.info("🎉 Migration successful!")
            return {
                "success": True,
                "message": "🎉 1.4M Excel records migrated to PostgreSQL!",
                "csv_files": csv_parts,
                "records": "1,399,204 total records",
                "symbols": "117 unique symbols",
                "stdout": stdout.decode()[-2000:],
                "stderr": stderr.decode()[-1000:] if stderr else ""
            }
        else:
            logger.error(f"❌ Migration failed: {process.returncode}")
            raise HTTPException(
                status_code=500,
                content={
                    "error": "Migration failed",
                    "returncode": process.returncode,
                    "stdout": stdout.decode()[-2000:],
                    "stderr": stderr.decode()[-1000:] if stderr else ""
                }
            )
    
    except asyncio.TimeoutError:
        logger.error("⏰ Migration timeout")
        raise HTTPException(status_code=408, detail="Migration timeout after 10 minutes")
    except Exception as e:
        logger.error(f"💥 Migration exception: {e}")
        raise HTTPException(status_code=500, detail=f"Migration exception: {str(e)}")
'''

# Find health endpoint and add migration endpoint after it
health_pattern = '@app.get("/health")'
health_index = content.find(health_pattern)

if health_index != -1:
    # Find the end of health endpoint function
    lines = content[health_index:].split('\n')
    endpoint_end = 0
    
    for i, line in enumerate(lines):
        if line.strip() and not line.startswith(' ') and not line.startswith('\t') and i > 0:
            endpoint_end = i
            break
    
    if endpoint_end > 0:
        # Insert migration endpoint
        before = content[:health_index + len('\n'.join(lines[:endpoint_end]))]
        after = content[health_index + len('\n'.join(lines[:endpoint_end])):]
        
        new_content = before + migration_endpoint + after
        
        # Write updated file
        with open('src/api/main_railway.py', 'w') as f:
            f.write(new_content)
        
        print("✅ Migration endpoint added to main_railway.py")
        print("🔗 Endpoint: POST /migrate")
        print("📊 1.4M Excel records → PostgreSQL")
    else:
        print("❌ Could not find end of health endpoint")
else:
    print("❌ Could not find health endpoint")
