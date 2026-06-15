import os
import time
import json
import threading
import requests
from llama_cpp import Llama

import signal
import sys

# Environment Variables
WORKER_NAME = os.environ.get("WORKER_NAME", "Unknown_Worker")
SUPPORTED_MODELS = os.environ.get("SUPPORTED_MODELS", "test-model").split(",")
NEXUS_URL = os.environ.get("NEXUS_URL", "http://127.0.0.1:10001")
MODEL_PATH = os.environ.get("MODEL_PATH", "/models/test.gguf")
N_GPU_LAYERS = int(os.environ.get("N_GPU_LAYERS", "0"))
N_CTX = int(os.environ.get("N_CTX", "2048"))
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
SPECS_JSON = os.environ.get("SPECS", '{"type": "generic"}')

# Global variables
WORKER_ID = None
CURRENT_STATUS = "ONLINE"
llm = None

shutdown_requested = False
is_processing = False

def handle_shutdown(signum, frame):
    global shutdown_requested, CURRENT_STATUS
    print(f"\n[{WORKER_NAME}] 🛑 Shutdown signal ({signum}) received. Initiating graceful shutdown...")
    shutdown_requested = True
    
    # Immediately notify the router that we are offline/shutting down
    CURRENT_STATUS = "OFFLINE"
    if WORKER_ID and NEXUS_URL:
        try:
            print(f"[{WORKER_NAME}] Informing Nexus that we are going OFFLINE...")
            requests.post(f"{NEXUS_URL}/worker/heartbeat", json={"worker_id": WORKER_ID, "status": "OFFLINE"}, timeout=2)
        except Exception as e:
            print(f"[{WORKER_NAME}] Failed to send offline status: {e}")
            
    # If we are not currently processing a task, exit immediately
    if not is_processing:
        print(f"[{WORKER_NAME}] No active tasks. Exiting now.")
        sys.exit(0)
    else:
        print(f"[{WORKER_NAME}] Currently processing a task. Will exit after completion...")

class MockLlama:
    def __init__(self, model_path):
        self.model_path = model_path
        
    def __call__(self, prompt, max_tokens=512, temperature=0.7, stream=False, **kwargs):
        import time
        response_text = f"Hello! This is a mock response from {WORKER_NAME} on CPU fallback. Prompt received: '{prompt}'."
        if stream:
            words = response_text.split(" ")
            chunks = []
            for i, word in enumerate(words):
                space = " " if i > 0 else ""
                chunks.append({
                    "choices": [{
                        "text": space + word
                    }]
                })
            
            class StreamIter:
                def __init__(self, items):
                    self.items = items
                def __iter__(self):
                    return self
                def __next__(self):
                    if not self.items:
                        raise StopIteration
                    time.sleep(0.1) # Simulate token generation delay
                    return self.items.pop(0)
            return StreamIter(chunks)
        else:
            time.sleep(0.5)
            return {
                "choices": [{
                    "text": response_text
                }]
            }

def load_model():
    print(f"[{WORKER_NAME}] Loading model from {MODEL_PATH} with n_gpu_layers={N_GPU_LAYERS}...")
    try:
        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=N_GPU_LAYERS,
            n_ctx=N_CTX,
            verbose=False
        )
        print(f"[{WORKER_NAME}] Model loaded successfully!")
        return llm
    except Exception as e:
        print(f"[{WORKER_NAME}] Failed to load model: {e}")
        print(f"[{WORKER_NAME}] ⚠️ Falling back to Mock LLM Engine for testing.")
        return MockLlama(MODEL_PATH)

def register_worker():
    global WORKER_ID, CURRENT_STATUS
    url = f"{NEXUS_URL}/worker/register"
    payload = {
        "worker_name": WORKER_NAME,
        "supported_models": SUPPORTED_MODELS,
        "specs": json.loads(SPECS_JSON)
    }
    while True:
        try:
            print(f"[{WORKER_NAME}] Registering with Nexus at {NEXUS_URL}...")
            resp = requests.post(url, json=payload, timeout=5)
            if resp.status_code == 200:
                WORKER_ID = resp.json().get("worker_id")
                CURRENT_STATUS = "ONLINE"
                print(f"[{WORKER_NAME}] Registered successfully! ID: {WORKER_ID} (Models: {SUPPORTED_MODELS})")
                break
        except Exception as e:
            print(f"[{WORKER_NAME}] Cannot reach Nexus: {e}. Retrying in 5s...")
        time.sleep(5)

def heartbeat_loop():
    url = f"{NEXUS_URL}/worker/heartbeat"
    while True:
        if WORKER_ID:
            try:
                requests.post(url, json={"worker_id": WORKER_ID, "status": CURRENT_STATUS}, timeout=2)
            except:
                pass
        time.sleep(5)

def poll_and_process():
    global llm, MODEL_PATH, SUPPORTED_MODELS, CURRENT_STATUS, is_processing
    poll_url = f"{NEXUS_URL}/worker/poll_task"
    stream_url = f"{NEXUS_URL}/worker/stream_chunk"
    complete_url = f"{NEXUS_URL}/worker/complete_task"
    
    print(f"[{WORKER_NAME}] Waiting for tasks...")
    while not shutdown_requested:
        try:
            resp = requests.get(poll_url, params={"worker_id": WORKER_ID}, timeout=5)
            if resp.status_code == 200:
                data = resp.json()
                
                # Check for admin commands (Hot-Swap)
                if data.get("command"):
                    cmd = data["command"]
                    if cmd.get("action") == "hotswap":
                        print(f"\n[{WORKER_NAME}] 🔥 Received HOT-SWAP Command! Setting status to UPDATING.")
                        CURRENT_STATUS = "UPDATING"
                        # Force heartbeat now to let Nexus know immediately
                        requests.post(f"{NEXUS_URL}/worker/heartbeat", json={"worker_id": WORKER_ID, "status": CURRENT_STATUS})
                        
                        print(f"[{WORKER_NAME}] Unloading old model to free VRAM...")
                        del llm
                        import gc
                        gc.collect()
                        
                        MODEL_PATH = cmd["new_model_path"]
                        SUPPORTED_MODELS = [cmd["new_alias"]]
                        llm = load_model()
                        
                        register_worker()
                        continue
                        
                if data.get("has_task"):
                    task_id = data["task_id"]
                    prompt = data["prompt"]
                    is_stream = data["stream_mode"]
                    
                    print(f"\n[{WORKER_NAME}] Received Task {task_id[:8]} (Stream: {is_stream})")
                    print(f"Prompt: {prompt}")
                    
                    is_processing = True
                    if shutdown_requested:
                        is_processing = False
                        break
                        
                    full_result = ""
                    try:
                        # Generate
                        response = llm(
                            prompt,
                            max_tokens=512,
                            temperature=TEMPERATURE,
                            stream=is_stream
                        )
                        
                        if is_stream:
                            for chunk in response:
                                text_chunk = chunk["choices"][0]["text"]
                                full_result += text_chunk
                                # Send chunk to nexus
                                requests.post(stream_url, json={
                                    "worker_id": WORKER_ID,
                                    "task_id": task_id,
                                    "chunk": text_chunk
                                })
                        else:
                            full_result = response["choices"][0]["text"]
                        
                        print(f"[{WORKER_NAME}] Completed Task {task_id[:8]}")
                        
                        # Submit complete
                        requests.post(complete_url, json={
                            "worker_id": WORKER_ID,
                            "task_id": task_id,
                            "result_content": full_result
                        })
                    finally:
                        is_processing = False
                        
                    if shutdown_requested:
                        print(f"[{WORKER_NAME}] Task completed during graceful shutdown. Exiting.")
                        break
                    
        except requests.exceptions.RequestException:
            pass # Nexus might be down temporarily
        except Exception as e:
            print(f"[{WORKER_NAME}] Error processing task: {e}")
            
        time.sleep(1)

if __name__ == "__main__":
    # Register signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, handle_shutdown)
    signal.signal(signal.SIGTERM, handle_shutdown)
    
    llm = load_model()
    register_worker()
    
    hb_thread = threading.Thread(target=heartbeat_loop, daemon=True)
    hb_thread.start()
    
    poll_and_process()
