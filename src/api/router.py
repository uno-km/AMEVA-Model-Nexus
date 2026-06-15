import asyncio
import uuid
import time
import json
import logging
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import aiohttp

from src.core.database import DatabaseManager, setup_db

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("NexusRouter")

# In-memory dictionary to hold asyncio queues for streaming tasks
# Format: { task_id: asyncio.Queue() }
STREAM_QUEUES = {}

# In-memory dictionary for pending admin commands to workers
# Format: { worker_id: {"action": "hotswap", ...} }
PENDING_COMMANDS = {}

# We will send logs to the Log Server (port 10003)
LOG_SERVER_URL = "http://127.0.0.1:10003/log/push"

async def push_log_to_server(source: str, level: str, payload_data: dict):
    try:
        async with aiohttp.ClientSession() as session:
            log_payload = {
                "source": source,
                "level": level,
                "payload": payload_data
            }
            async with session.post(LOG_SERVER_URL, json=log_payload, timeout=2) as resp:
                await resp.read()
    except Exception as e:
        logger.error(f"Failed to push log: {e}")

# --- Background Tasks ---

async def dispatcher_loop():
    """Continuously assigns PENDING tasks to ONLINE workers"""
    logger.info("[Dispatcher] Started")
    while True:
        try:
            pending_tasks = DatabaseManager.router_get_pending_tasks()
            if pending_tasks:
                workers = DatabaseManager.router_get_workers()
                online_workers = [w for w in workers if w['status'] == 'ONLINE']
                
                for task in pending_tasks:
                    task_id = task['task_id']
                    req_model = task['model_name']
                    
                    suitable_worker = next((w for w in online_workers if req_model in w['supported_models']), None)
                    if suitable_worker:
                        worker_id = suitable_worker['worker_id']
                        DatabaseManager.router_assign_task(task_id, worker_id)
                        DatabaseManager.router_set_worker_status(worker_id, "BUSY")
                        online_workers.remove(suitable_worker)
                        logger.info(f"[Dispatcher] Task {task_id[:8]} -> Worker {worker_id[:8]}")
        except Exception as e:
            logger.error(f"[Dispatcher] Error: {e}")
        
        await asyncio.sleep(1)

async def watchdog_loop():
    """Monitors for crashed workers and bottlenecked tasks"""
    logger.info("[Watchdog] Started")
    offline_timeout = 15.0
    task_timeout = 300.0  # 5 minutes for real LLM generation

    while True:
        try:
            now = datetime.now()
            
            # 1. Offline Timeout
            workers = DatabaseManager.router_get_workers()
            for w in workers:
                if w['last_heartbeat']:
                    last_hb = datetime.fromisoformat(w['last_heartbeat'])
                    diff_sec = (now - last_hb).total_seconds()
                    if diff_sec > offline_timeout and w['status'] != 'OFFLINE':
                        logger.warning(f"[Watchdog] Worker {w['worker_name']} offline ({diff_sec:.1f}s)")
                        DatabaseManager.router_set_worker_status(w['worker_id'], 'OFFLINE')

            # 2. Bottleneck Timeout & Crash Re-queue
            processing_tasks = DatabaseManager.router_get_processing_tasks()
            for task in processing_tasks:
                started_at = datetime.fromisoformat(task['started_at'])
                worker_id = task['assigned_worker_id']
                diff_sec = (now - started_at).total_seconds()
                
                assigned_worker = next((w for w in workers if w['worker_id'] == worker_id), None)
                
                # If assigned worker crashed
                if assigned_worker and assigned_worker['status'] == 'OFFLINE':
                    logger.warning(f"[Watchdog] Task {task['task_id'][:8]} assigned to OFFLINE worker. Re-queueing.")
                    DatabaseManager.router_requeue_task(task['task_id'])
                    # Tell stream queue if it exists
                    q = STREAM_QUEUES.get(task['task_id'])
                    if q: await q.put("[WARN: Worker crashed, task re-queued. Please wait...]\n")
                    continue
                
                # Bottleneck
                if diff_sec > task_timeout:
                    logger.error(f"[Watchdog] Task {task['task_id'][:8]} bottlenecked ({diff_sec:.1f}s). Cancelling.")
                    worker_specs = assigned_worker['specs'] if assigned_worker else "{}"
                    
                    DatabaseManager.router_log_bottleneck(
                        task_id=task['task_id'],
                        worker_id=worker_id,
                        prompt=task['prompt'],
                        model_name=task['model_name'],
                        worker_specs=worker_specs,
                        processing_time_sec=diff_sec
                    )
                    DatabaseManager.router_fail_task(task['task_id'], new_status="TIMEOUT")
                    
                    if assigned_worker and assigned_worker['status'] == 'BUSY':
                        DatabaseManager.router_set_worker_status(worker_id, "ONLINE")
                        
                    q = STREAM_QUEUES.get(task['task_id'])
                    if q: await q.put("[ERROR: Task timed out]\n[DONE]")

        except Exception as e:
            logger.error(f"[Watchdog] Error: {e}")
            
        await asyncio.sleep(5)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_db()
    
    # Start background loops
    task1 = asyncio.create_task(dispatcher_loop())
    task2 = asyncio.create_task(watchdog_loop())
    
    yield
    
    task1.cancel()
    task2.cancel()


app = FastAPI(title="AMEVA Model Nexus Router", lifespan=lifespan)

# --- Pydantic Models ---

class ChatRequest(BaseModel):
    model: str
    prompt: str
    stream: bool = False

class WorkerRegisterReq(BaseModel):
    worker_name: str
    supported_models: list[str]
    specs: dict

class WorkerHeartbeatReq(BaseModel):
    worker_id: str
    status: str = "ONLINE"

class StreamChunkReq(BaseModel):
    worker_id: str
    task_id: str
    chunk: str

class CompleteTaskReq(BaseModel):
    worker_id: str
    task_id: str
    result_content: str

class HotSwapReq(BaseModel):
    target_worker: str
    new_model_path: str
    new_alias: str

# --- User/Client API ---

@app.get("/help")
async def get_help():
    """Returns documentation for the API."""
    workers = DatabaseManager.router_get_workers()
    online_workers = [w for w in workers if w['status'] == 'ONLINE']
    
    available_models = set()
    for w in online_workers:
        models = w['supported_models'].split(',')
        for m in models:
            if m.strip(): available_models.add(m.strip())
            
    help_info = {
        "api_description": "AMEVA Model Nexus - Production API Gateway",
        "endpoints": {
            "POST /api/chat": "Submit a request to an LLM.",
            "POST /admin/hotswap": "Zero-downtime hot-swap a model on a worker.",
            "GET /help": "This documentation."
        },
        "chat_parameters": {
            "model": "String. Required. Example: 'Llama-8'",
            "prompt": "String. Required. Your input text.",
            "stream": "Boolean. Default: false. If true, returns Server-Sent Events (real-time typing)."
        },
        "hotswap_parameters": {
            "target_worker": "String. The worker_name or worker_id.",
            "new_model_path": "String. Absolute path inside the worker container/machine.",
            "new_alias": "String. The new simple alias like 'Llama-8-v2'."
        },
        "available_models_currently_online": list(available_models),
        "how_streaming_works": "When stream=true, you will receive text chunks as they are generated by the worker. The stream ends with the exact string '[DONE]'."
    }
    return JSONResponse(content=help_info)

@app.post("/admin/hotswap")
async def admin_hotswap(req: HotSwapReq):
    # Find worker ID by name or exact ID
    workers = DatabaseManager.router_get_workers()
    target = next((w for w in workers if w['worker_name'] == req.target_worker or w['worker_id'] == req.target_worker), None)
    
    if not target:
        raise HTTPException(status_code=404, detail="Worker not found.")
        
    worker_id = target['worker_id']
    PENDING_COMMANDS[worker_id] = {
        "action": "hotswap",
        "new_model_path": req.new_model_path,
        "new_alias": req.new_alias
    }
    return {"status": "ok", "message": f"Hot-swap command queued for worker {target['worker_name']} ({worker_id}). Queueing requests until ready."}

@app.post("/api/chat")
async def api_chat(req: ChatRequest, request: Request):
    client_ip = request.client.host
    model = req.model
    
    # 1. Model Availability Check
    workers = DatabaseManager.router_get_workers()
    online_workers = [w for w in workers if w['status'] == 'ONLINE']
    
    available_models = set()
    for w in online_workers:
        models = w['supported_models'].split(',')
        for m in models:
            if m.strip(): available_models.add(m.strip())
            
    if model not in available_models:
        raise HTTPException(
            status_code=500, 
            detail=f"Model '{model}' is not supported by any currently online workers. Available models: {list(available_models)}"
        )

    # 2. Create Task
    task_id = DatabaseManager.router_create_task(
        model_name=model, 
        prompt=req.prompt, 
        client_ip=client_ip, 
        stream_mode=req.stream
    )
    
    logger.info(f"[API] Task {task_id[:8]} created by {client_ip} for {model}")

    # 3. Handle Streaming
    if req.stream:
        STREAM_QUEUES[task_id] = asyncio.Queue()
        
        async def event_generator():
            full_content = ""
            queue = STREAM_QUEUES[task_id]
            try:
                while True:
                    chunk = await queue.get()
                    if chunk == "[DONE]":
                        break
                    
                    # Also append to a local string buffer so we can log it at the end
                    if not chunk.startswith("[WARN:") and not chunk.startswith("[ERROR:"):
                        full_content += chunk
                        
                    yield f"data: {json.dumps({'chunk': chunk})}\n\n"
                    
            except asyncio.CancelledError:
                logger.warning(f"Client disconnected stream for task {task_id[:8]}")
            finally:
                # When stream is done or client disconnects, send full log to central DB
                await push_log_to_server(
                    source="NexusAPI",
                    level="INFO",
                    payload_data={
                        "event": "Stream_Finished",
                        "task_id": task_id,
                        "client_ip": client_ip,
                        "model": model,
                        "prompt_preview": req.prompt[:100],
                        "final_result": full_content
                    }
                )
                if task_id in STREAM_QUEUES:
                    del STREAM_QUEUES[task_id]
                    
        return StreamingResponse(event_generator(), media_type="text/event-stream")

    # 4. Handle Static Mode (Wait for complete)
    else:
        while True:
            task = DatabaseManager.router_get_task(task_id)
            if task and task['status'] == 'COMPLETED':
                result = task['result_content']
                # Log to central DB
                await push_log_to_server(
                    source="NexusAPI",
                    level="INFO",
                    payload_data={
                        "event": "Static_Finished",
                        "task_id": task_id,
                        "client_ip": client_ip,
                        "model": model,
                        "prompt_preview": req.prompt[:100],
                        "final_result": result
                    }
                )
                return {"task_id": task_id, "result": result}
            elif task and task['status'] in ['FAILED', 'TIMEOUT']:
                raise HTTPException(status_code=500, detail=f"Task failed or timed out: {task['status']}")
            
            await asyncio.sleep(0.5)

# --- Worker API ---

@app.post("/worker/register")
async def worker_register(req: WorkerRegisterReq):
    worker_id = str(uuid.uuid4())
    DatabaseManager.router_upsert_worker(
        worker_id=worker_id,
        worker_name=req.worker_name,
        supported_models=",".join(req.supported_models),
        specs=json.dumps(req.specs)
    )
    return {"worker_id": worker_id}

@app.post("/worker/heartbeat")
async def worker_heartbeat(req: WorkerHeartbeatReq):
    DatabaseManager.router_update_worker_heartbeat(req.worker_id, req.status)
    return {"status": "ok"}

@app.get("/worker/poll_task")
async def worker_poll(worker_id: str):
    # 1. Check for admin commands
    if worker_id in PENDING_COMMANDS:
        cmd = PENDING_COMMANDS.pop(worker_id)
        return {"has_task": False, "command": cmd}

    # 2. Find if any task is ASSIGNED to this worker (status = PROCESSING)
    tasks = DatabaseManager.router_get_processing_tasks()
    my_tasks = [t for t in tasks if t['assigned_worker_id'] == worker_id]
    
    if my_tasks:
        # Worker might have crashed and restarted, give it the task it's supposed to do
        t = my_tasks[0]
        # Return task info
        return {"has_task": True, "task_id": t['task_id'], "prompt": t['prompt'], "model": t['model_name'], "stream_mode": t['stream_mode']}
    
    return {"has_task": False}

@app.post("/worker/stream_chunk")
async def worker_stream_chunk(req: StreamChunkReq):
    # Route chunk to the appropriate streaming queue
    q = STREAM_QUEUES.get(req.task_id)
    if q:
        await q.put(req.chunk)
    return {"status": "ok"}

@app.post("/worker/complete_task")
async def worker_complete(req: CompleteTaskReq):
    DatabaseManager.router_complete_task(req.task_id, req.result_content)
    DatabaseManager.router_set_worker_status(req.worker_id, "ONLINE")
    
    # Send [DONE] signal to stream if applicable
    q = STREAM_QUEUES.get(req.task_id)
    if q:
        await q.put("[DONE]")
        
    return {"status": "ok"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10001)
