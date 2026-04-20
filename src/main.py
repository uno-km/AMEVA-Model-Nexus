# src/main.py
import os
import time
import json
import gzip
import shutil
import psutil
import asyncio
import threading
from collections import deque
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, TYPE_CHECKING

import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# 실행 방식에 따라 import 경로 달라질 수 있으니 안전 처리
try:
    from src.pc_launcher import AmevaPCNodeLauncher
except Exception:
    from pc_launcher import AmevaPCNodeLauncher

if TYPE_CHECKING:
    from llama_cpp import Llama  # PC에서는 절대 import 안 됨 (타입체크 전용)

app = FastAPI(title="AMEVA Model Nexus (Auto-Scaling)")


# =========================================================
# Env / Config
# =========================================================

def detect_environment() -> str:
    prefix = os.environ.get("PREFIX", "")
    if "com.termux" in prefix or os.path.exists("/data/data/com.termux"):
        return "MOBILE"
    return "PC"


ENV_TYPE = detect_environment()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.json")


def load_config(path: str) -> dict:
    if not os.path.exists(path):
        raise RuntimeError(f"config.json not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


CONFIG = load_config(CONFIG_PATH)


def resolve_node_key() -> str:
    # PC는 EXPERT_PC 고정
    if ENV_TYPE == "PC":
        return "EXPERT_PC"
    # 모바일은 환경변수로 노드 선택 가능: ROUTER / EXPERT_MOBILE
    return os.environ.get("AMEVA_NODE", "ROUTER")


NODE_KEY = resolve_node_key()
NODE_CFG = CONFIG["nodes"][NODE_KEY]

# Hub 포트: 13000대
HUB_PORT = 13000 if ENV_TYPE == "PC" else 13001

# PC Docker llama.cpp 포트: 14000대 (config)
LLM_PORT = int(NODE_CFG.get("port", 14000))
LLM_BASE_URL = f"http://127.0.0.1:{LLM_PORT}"

MODEL_DIR = NODE_CFG.get("model_dir", os.path.join(PROJECT_ROOT, "models"))
MODEL_FILENAME = NODE_CFG.get("model_filename", "qwen-3b.gguf")
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_FILENAME)

LLM_CTX_SIZE = int(NODE_CFG.get("ctx_size", 2048))
LLM_THREADS = int(NODE_CFG.get("threads", 4))
RAM_LIMIT_PERCENT = float(NODE_CFG.get("ram_limit", 93.0))

MAX_CONCURRENT_CHUNKS = 1 if ENV_TYPE == "MOBILE" else max(1, min(8, LLM_THREADS))
INFERENCE_TIMEOUT = 180.0 if ENV_TYPE == "MOBILE" else 30.0


# =========================================================
# Globals
# =========================================================

llm: Optional["Llama"] = None  # MOBILE only
launcher: Optional[AmevaPCNodeLauncher] = None  # PC only

executor = ThreadPoolExecutor(max_workers=MAX_CONCURRENT_CHUNKS)
concurrency_semaphore = asyncio.Semaphore(MAX_CONCURRENT_CHUNKS)


# =========================================================
# Monitor / Logger
# =========================================================

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
                self.cpu = 50.0
                self.ram = 50.0

            timestamp = datetime.now().strftime("%H:%M:%S")
            self.history.append({"time": timestamp, "cpu": self.cpu, "ram": self.ram})

            await asyncio.sleep(2.0 if ENV_TYPE == "MOBILE" else 1.0)

    @property
    def avg_inference_time(self):
        if self.completed_chunks == 0:
            return 0.0
        return round(self.total_inference_time / self.completed_chunks, 2)


sys_monitor = SystemMonitor()


class ElasticLogger:
    def __init__(self):
        self.heap_buffer = []
        self.lock = threading.Lock()
        self.log_file = os.path.join(PROJECT_ROOT, "AMEVA_Universal_Log.txt")
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
            with open(filename, "rb") as f_in, gzip.open(f"{filename}.gz", "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
            os.remove(filename)
        except Exception:
            pass

    async def flush_worker(self):
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

                try:
                    if os.path.getsize(self.log_file) > self.max_bytes:
                        rotated = os.path.join(
                            PROJECT_ROOT,
                            f"AMEVA_Log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
                        )
                        os.rename(self.log_file, rotated)
                        loop = asyncio.get_running_loop()
                        loop.run_in_executor(None, self._compress_file, rotated)
                except Exception:
                    pass


elastic_logger = ElasticLogger()


# =========================================================
# Prompt formatting ("이쁘게 만들기")
# =========================================================

def format_prompt(user_text: str) -> str:
    """
    니가 말한 '프롬프트 이쁘게 만들기'를 한 곳에서 통일.
    추후 ChatML/템플릿으로 갈아끼우기 쉽게 만든 함수.

    - 지금은 기본 system prompt + user prompt 구조로만 감싼다.
    - 나중에 Qwen ChatML로 바꾸려면 여기만 수정하면 됨.
    """
    system_prompt = CONFIG.get("system_prompt", "You are a helpful assistant.")
    user_text = (user_text or "").strip()

    # 아주 단순한 템플릿
    return f"System: {system_prompt}\nUser: {user_text}\nAssistant:"


# =========================================================
# Throttling
# =========================================================

async def wait_for_greenlight(max_wait_sec: float = 20.0):
    start = time.time()
    while True:
        if sys_monitor.cpu < 90 and sys_monitor.ram < RAM_LIMIT_PERCENT:
            return sys_monitor.cpu, sys_monitor.ram
        if time.time() - start > max_wait_sec:
            raise HTTPException(status_code=503, detail="Server overloaded (resource throttle)")
        await asyncio.sleep(1.0)


# =========================================================
# PC backend self-healing (optional but recommended)
# =========================================================

def ensure_pc_backend_ready():
    """
    PC 도커 백엔드가 죽었을 때 자동으로 다시 launch 하도록.
    """
    global launcher
    if launcher is None:
        launcher = AmevaPCNodeLauncher(config_path=CONFIG_PATH, node_key="EXPERT_PC")

    # health check (있으면)
    try:
        r = requests.get(f"{LLM_BASE_URL}/health", timeout=2)
        if r.status_code == 200:
            return
    except Exception:
        pass

    # fallback: 재기동 시도
    ok = launcher.launch()
    if not ok:
        raise HTTPException(status_code=503, detail="LLM backend unavailable (docker)")


# =========================================================
# Inference functions (진짜 핵심)
# =========================================================

def infer_mobile(prompt: str) -> str:
    """
    MOBILE: llama_cpp 로컬 호출
    """
    global llm
    if llm is None:
        raise RuntimeError("LLM not loaded (mobile).")

    out = llm(prompt, max_tokens=512, echo=False)
    return out["choices"][0]["text"]


def infer_pc(prompt: str) -> str:
    """
    PC: 도커 llama.cpp server로 호출
    - 가능하면 pc_launcher에 infer_completion 만들어서 그걸 쓰는게 정석
    - 없으면 requests로 때린다.
    """
    global launcher
    ensure_pc_backend_ready()

    # 1) launcher에 infer_completion이 있으면 그걸 사용
    if launcher is not None and hasattr(launcher, "infer_completion"):
        return launcher.infer_completion(
            prompt,
            n_predict=512,
            temperature=0.3,
            top_p=0.7,
            timeout=INFERENCE_TIMEOUT
        )

    # 2) fallback: 직접 /completion 호출
    payload = {
        "prompt": prompt,
        "n_predict": 512,
        "temperature": 0.3,
        "top_p": 0.7,
        "stream": False
    }
    r = requests.post(f"{LLM_BASE_URL}/completion", json=payload, timeout=INFERENCE_TIMEOUT)
    r.raise_for_status()
    data = r.json()
    return data.get("content") or data.get("choices", [{}])[0].get("text", "")


# =========================================================
# Startup
# =========================================================

@app.on_event("startup")
async def startup_event():
    global llm, launcher

    asyncio.create_task(sys_monitor.monitor_loop())
    asyncio.create_task(elastic_logger.flush_worker())

    elastic_logger.add_log("SYSTEM", {
        "msg": "Hub starting",
        "env": ENV_TYPE,
        "node": NODE_KEY,
        "hub_port": HUB_PORT,
        "llm_port": LLM_PORT,
        "concurrency": MAX_CONCURRENT_CHUNKS
    })

    if ENV_TYPE == "PC":
        # PC는 도커만 올림
        launcher = AmevaPCNodeLauncher(config_path=CONFIG_PATH, node_key="EXPERT_PC")
        ok = launcher.launch()
        if not ok:
            raise RuntimeError("Failed to launch docker llama.cpp server.")
        llm = None  # PC에서는 llama_cpp 절대 사용 안 함

        elastic_logger.add_log("SYSTEM", {"msg": "Docker llama.cpp server ready", "base_url": LLM_BASE_URL})

    else:
        # MOBILE일 때만 llama_cpp import + load
        if not os.path.exists(MODEL_PATH):
            raise RuntimeError(f"Model file not found: {MODEL_PATH}")

        from llama_cpp import Llama  # ✅ MOBILE only lazy import

        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=0,
            n_ctx=LLM_CTX_SIZE,
            n_threads=LLM_THREADS
        )
        launcher = None
        elastic_logger.add_log("SYSTEM", {"msg": "Mobile llama_cpp loaded", "model": MODEL_PATH})


# =========================================================
# API
# =========================================================

@app.get("/health")
def health():
    return {"ok": True, "env": ENV_TYPE, "node": NODE_KEY}


@app.get("/stats")
def get_stats():
    return {
        "env_type": ENV_TYPE,
        "node": NODE_KEY,
        "hub_port": HUB_PORT,
        "llm_port": LLM_PORT,
        "active_chunks": sys_monitor.active_chunks,
        "completed_chunks": sys_monitor.completed_chunks,
        "avg_inference_time_sec": sys_monitor.avg_inference_time,
        "cpu_percent": sys_monitor.cpu,
        "ram_percent": sys_monitor.ram,
        "ram_limit_percent": RAM_LIMIT_PERCENT,
        "concurrency": MAX_CONCURRENT_CHUNKS
    }


def _is_model_file(filename: str) -> bool:
    allowed = {".gguf", ".bin", ".pth", ".pt", ".ckpt", ".safetensors", ".h5", ".pb", ".onnx", ".tflite"}
    return os.path.splitext(filename)[1].lower() in allowed


@app.get("/list")
def list_models():
    try:
        files = sorted(os.listdir(MODEL_DIR))
    except FileNotFoundError:
        return {"model_directory": MODEL_DIR, "models": []}

    model_files = [
        f for f in files
        if os.path.isfile(os.path.join(MODEL_DIR, f)) and _is_model_file(f)
    ]
    return {"model_directory": MODEL_DIR, "models": model_files}


class ChunkRequest(BaseModel):
    chunk_id: int
    text: str


@app.post("/generate/chunk")
async def generate_chunk(req: ChunkRequest):
    elastic_logger.add_log("RECV_CHUNK", {"chunk_id": req.chunk_id, "len": len(req.text)})

    async with concurrency_semaphore:
        cpu, ram = await wait_for_greenlight()
        sys_monitor.active_chunks += 1
        elastic_logger.add_log("START_INFERENCE", {"chunk_id": req.chunk_id, "cpu": cpu, "ram": ram})

        start_time = time.time()
        loop = asyncio.get_running_loop()

        # ✅ 여기서 "프롬프트 이쁘게 만들기"를 단 1번만 함
        formatted = format_prompt(req.text)

        try:
            if ENV_TYPE == "PC":
                # ✅ PC는 무조건 도커로만
                result_text = await asyncio.wait_for(
                    loop.run_in_executor(executor, lambda: infer_pc(formatted)),
                    timeout=INFERENCE_TIMEOUT
                )
            else:
                # ✅ MOBILE은 무조건 로컬 llama_cpp로만
                result_text = await asyncio.wait_for(
                    loop.run_in_executor(executor, lambda: infer_mobile(formatted)),
                    timeout=INFERENCE_TIMEOUT
                )

            elapsed = time.time() - start_time
            sys_monitor.total_inference_time += elapsed
            sys_monitor.completed_chunks += 1

            elastic_logger.add_log("DONE_CHUNK", {"chunk_id": req.chunk_id, "time": round(elapsed, 2)})
            return {"chunk_id": req.chunk_id, "result": result_text}

        except asyncio.TimeoutError:
            elastic_logger.add_log("TIMEOUT_CHUNK", {"chunk_id": req.chunk_id})
            raise HTTPException(status_code=504, detail=f"{ENV_TYPE} Inference Timeout")

        except HTTPException:
            raise

        except Exception as e:
            elastic_logger.add_log("ERROR_CHUNK", {"chunk_id": req.chunk_id, "error": str(e)})
            raise HTTPException(status_code=500, detail=str(e))

        finally:
            sys_monitor.active_chunks -= 1


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=HUB_PORT)