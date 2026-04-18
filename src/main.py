import os
import time
from pc_launcher import AmevaPCNodeLauncher
import json
import gzip
import shutil
import psutil
import asyncio
import threading
from collections import deque
from datetime import datetime
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from llama_cpp import Llama
from concurrent.futures import ThreadPoolExecutor

app = FastAPI(title="AMEVA Universal Hub (Auto-Scaling)")

# ==========================================
# 🕵️‍♂️ 환경 탐지기 (Environment Detector)
# ==========================================
def detect_environment():
    """현재 실행 환경이 안드로이드(Termux)인지, PC(Windows/Linux)인지 판별합니다."""
    # 터뮤즈 전용 환경 변수 확인
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" in prefix or os.path.exists("/data/data/com.termux"):
        return "MOBILE"
    return "PC"

ENV_TYPE = detect_environment()

# (팁) 만약 모델 폴더를 통일하고 싶다면 현재 폴더 기준 상대경로("./models/qwen.gguf")를 쓰시면 더 완벽합니다.
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")

llm = None
executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHUNKS)
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)

# ==========================================
# 🧠 하이브리드 자원 모니터 (PC/Mobile 겸용)
# ==========================================
class SystemMonitor:
    def __init__(self):
        self.cpu = 0.0
        self.ram = 0.0
        self.history = deque(maxlen=60)
        self.active_chunks = 0
        self.completed_chunks = 0
        self.total_inference_time = 0.0

    async def monitor_loop(self):
        while True:
            try:
                self.cpu = psutil.cpu_percent(interval=None)
                self.ram = psutil.virtual_memory().percent
            except Exception:
                # 터뮤즈 권한 에러 시 시스템 동결 방지용 가짜 값
                self.cpu = 50.0
                self.ram = 50.0
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            self.history.append({"time": timestamp, "cpu": self.cpu, "ram": self.ram})
            
            # 모바일은 배터리 보호를 위해 2초 단위, PC는 1초 단위로 측정
            sleep_time = 2.0 if ENV_TYPE == "MOBILE" else 1.0
            await asyncio.sleep(sleep_time)

    @property
    def avg_inference_time(self):
        if self.completed_chunks == 0: return 0.0
        return round(self.total_inference_time / self.completed_chunks, 2)

sys_monitor = SystemMonitor()

# ==========================================
# 📊 엘라스틱 로거 (스마트 용량 조절)
# ==========================================
class ElasticLogger:
    def __init__(self):
        self.heap_buffer = []
        self.lock = threading.Lock()
        self.log_file = "AMEVA_Universal_Log.txt"
        # 모바일은 50MB, PC는 100MB에서 로그 압축(Rotation)
        self.max_bytes = (50 if ENV_TYPE == "MOBILE" else 100) * 1024 * 1024 

    def add_log(self, event_type: str, data: dict):
        with self.lock:
            self.heap_buffer.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "event": event_type,
                **data
            })

    def _compress_file(self, filename: str):
        try:
            with open(filename, 'rb') as f_in, gzip.open(f"{filename}.gz", 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(filename)
        except Exception:
            pass

    async def flush_worker(self):
        # 디스크 I/O 주기 (모바일 10초, PC 5초)
        flush_interval = 10 if ENV_TYPE == "MOBILE" else 5
        while True:
            await asyncio.sleep(flush_interval)
            logs_to_write = []
            with self.lock:
                if self.heap_buffer:
                    logs_to_write = self.heap_buffer.copy()
                    self.heap_buffer.clear()
            
            if logs_to_write:
                with open(self.log_file, "a", encoding="utf-8") as f:
                    for log in logs_to_write:
                        f.write(json.dumps(log, ensure_ascii=False) + "\n")
                
                if os.path.getsize(self.log_file) > self.max_bytes:
                    rotated_file = f"AMEVA_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                    os.rename(self.log_file, rotated_file)
                    loop = asyncio.get_running_loop()
                    loop.run_in_executor(None, self._compress_file, rotated_file)

elastic_logger = ElasticLogger()

async def wait_for_greenlight():
    """모바일과 PC의 각기 다른 RAM 임계치(85% vs 93%)를 동적으로 적용"""
    while True:
        if sys_monitor.cpu < 90 and sys_monitor.ram < RAM_LIMIT_PERCENT:
            return sys_monitor.cpu, sys_monitor.ram
        await asyncio.sleep(1.0)

# ==========================================
# 📡 라이프사이클 및 API 엔드포인트
# ==========================================
@app.on_event("startup")
async def startup_event():
    global llm
    
    # 기기 스펙에 맞춰진 동적 옵션으로 Llama 엔진 로드
    llm = Llama(
        model_path=MODEL_PATH, 
        n_gpu_layers=LLM_GPU_LAYERS,
        n_ctx=LLM_CTX_SIZE,
        n_threads=4 if ENV_TYPE == "MOBILE" else None # 모바일은 4코어로 제한, PC는 자동 할당
    )
    
    asyncio.create_task(sys_monitor.monitor_loop())
    asyncio.create_task(elastic_logger.flush_worker())
    elastic_logger.add_log("SYSTEM", {"msg": f"AMEVA {ENV_TYPE} Hub Started (Concurrency: {MAX_CONCURRENT_CHUNKS})"})

@app.get("/stats")
def get_stats():
    return {
        "device_type": ENV_TYPE,
        "active_chunks": sys_monitor.active_chunks,
        "completed_chunks": sys_monitor.completed_chunks,
        "avg_inference_time_sec": sys_monitor.avg_inference_time,
        "cpu_percent": sys_monitor.cpu,
        "ram_percent": sys_monitor.ram
    }

def _is_model_file(filename: str) -> bool:
    allowed_exts = {".gguf", ".bin", ".pth", ".pt", ".ckpt", ".safetensors", ".h5", ".pb", ".onnx", ".tflite"}
    return os.path.splitext(filename)[1].lower() in allowed_exts

@app.get("/list")
def list_models():
    """models 폴더 내 모델 파일 목록을 반환합니다."""
    try:
        files = sorted(os.listdir(MODEL_DIR))
    except FileNotFoundError:
        return {"model_directory": MODEL_DIR, "models": []}

    model_files = [f for f in files if os.path.isfile(os.path.join(MODEL_DIR, f)) and _is_model_file(f)]
    return {"model_directory": MODEL_DIR, "models": model_files}

class ChunkRequest(BaseModel):
    chunk_id: int
    text: str

@app.post("/generate/chunk")
async def generate_chunk(req: ChunkRequest):
    elastic_logger.add_log("RECV_CHUNK", {"chunk_id": req.chunk_id, "len": len(req.text)})

    # 동적으로 설정된 n차선 톨게이트 (PC는 8, 모바일은 1)
    async with concurrency_semaphore:
        cpu, ram = await wait_for_greenlight()
        sys_monitor.active_chunks += 1
        elastic_logger.add_log("START_INFERENCE", {"chunk_id": req.chunk_id})
        
        start_time = time.time()
        loop = asyncio.get_running_loop()
        try:
            # 타임아웃 역시 동적 적용 (PC 30초, 모바일 180초)
            output = await asyncio.wait_for(
                loop.run_in_executor(executor, lambda: llm(req.text, max_tokens=512, echo=False)),
                timeout=INFERENCE_TIMEOUT
            )
            result_text = output["choices"][0]["text"]
            
            elapsed = time.time() - start_time
            sys_monitor.total_inference_time += elapsed
            sys_monitor.completed_chunks += 1
            
            elastic_logger.add_log("DONE_CHUNK", {"chunk_id": req.chunk_id, "time": round(elapsed, 2)})
            return {"chunk_id": req.chunk_id, "result": result_text}
            
        except asyncio.TimeoutError:
            elastic_logger.add_log("TIMEOUT_CHUNK", {"chunk_id": req.chunk_id})
            raise HTTPException(status_code=504, detail=f"{ENV_TYPE} Inference Timeout")
        except Exception as e:
            elastic_logger.add_log("ERROR_CHUNK", {"chunk_id": req.chunk_id, "error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))
        finally:
            sys_monitor.active_chunks -= 1

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)