import os
import sqlite3
import asyncio
import json
from datetime import datetime
from typing import Any, Dict
from contextlib import asynccontextmanager

from src.core.database import DatabaseManager, get_connection

from fastapi import FastAPI, Request

DB_PATH = "ameva_universal_logs.db"
FLUSH_INTERVAL = 10.0
CHUNK_SIZE = 1000

# Create DB with WAL mode for extreme concurrency and IO performance
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA synchronous=NORMAL;")
    conn.execute("PRAGMA mmap_size=30000000000;")
    conn.execute('''
        CREATE TABLE IF NOT EXISTS universal_logs (
            log_id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            source TEXT,
            level TEXT,
            payload_json TEXT
        )
    ''')
    conn.commit()
    conn.close()

app = FastAPI(title="AMEVA 1MB-class Log Ingester")
log_queue = asyncio.Queue()

@app.on_event("startup")
async def startup_event():
    init_db()
    asyncio.create_task(db_flusher_loop())
    print(f"🚀 Log Server Started. Writing to {DB_PATH}")

async def db_flusher_loop():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    cursor = conn.cursor()
    
    while True:
        # 무조건 10초 대기
        await asyncio.sleep(FLUSH_INTERVAL)
        
        batch = []
        # 10초간 쌓인 큐를 한 번에 다 빼냅니다.
        while not log_queue.empty():
            batch.append(log_queue.get_nowait())
            log_queue.task_done()
            
        if not batch:
            continue # 쌓인 로그가 없으면 스킵
            
        print(f"[LogServer] ⏱️ 10초 주기 도달! 총 {len(batch)}건의 로그를 1,000건씩 촤르르륵 분할 저장합니다...")
        
        # 1000건씩 쪼개서 (chunk) executemany 실행
        total_chunks = (len(batch) + CHUNK_SIZE - 1) // CHUNK_SIZE
        for i in range(0, len(batch), CHUNK_SIZE):
            chunk = batch[i:i + CHUNK_SIZE]
            cursor.executemany('''
                INSERT INTO universal_logs (timestamp, source, level, payload_json)
                VALUES (?, ?, ?, ?)
            ''', chunk)
            conn.commit()
            print(f"   -> 💾 저장 완료: 청크 {(i//CHUNK_SIZE)+1}/{total_chunks} ({len(chunk)}건)")
            
            # 너무 혼자 독점하지 않고 비동기 이벤트 루프를 넘겨주면서 "촤르륵" 느낌을 위해 아주 짧게 양보
            await asyncio.sleep(0.01)

@app.post("/log/push")
async def push_log(req: Request):
    """
    Expects JSON:
    {
        "source": "worker_name",
        "level": "INFO/ERROR/WARN/BOTTLENECK",
        "payload": { anything }
    }
    """
    data = await req.json()
    source = data.get("source", "UNKNOWN")
    level = data.get("level", "INFO")
    payload = data.get("payload", {})
    
    timestamp = datetime.now().isoformat()
    payload_str = json.dumps(payload, ensure_ascii=False)
    
    # Put into memory queue immediately to return HTTP 200 instantly (non-blocking)
    await log_queue.put((timestamp, source, level, payload_str))
    
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    # Start the server on port 14003
    uvicorn.run(app, host="0.0.0.0", port=10003)
