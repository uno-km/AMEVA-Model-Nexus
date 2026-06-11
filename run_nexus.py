import os
import sys
import subprocess
import time

def check_nvidia_gpu():
    try:
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def check_cuda_env():
    # Check if CUDA_PATH is set (Official Windows standard for CUDA Toolkit)
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path and os.path.exists(cuda_path):
        return True
    return False

def get_llama_engine_status():
    try:
        from llama_cpp import llama_supports_gpu_offload
        return "GPU" if llama_supports_gpu_offload() else "CPU"
    except Exception as e:
        err_str = str(e).lower()
        if "llama.dll" in err_str or "cudart" in err_str:
            return "GPU_MISSING_CUDA"
        return "NONE"

def check_and_install_dependencies():
    print("=======================================")
    print(" 🚀 AMEVA Nexus Smart Launcher")
    print("=======================================")
    
    has_gpu = check_nvidia_gpu()
    has_cuda = check_cuda_env()

    if has_gpu:
        if has_cuda:
            print("[1/3] Hardware Scan: 🟢 NVIDIA GPU & CUDA_PATH Found! (Target: GPU Mode)")
        else:
            print("[1/3] Hardware Scan: 🟡 NVIDIA GPU Found, BUT missing CUDA Toolkit (CUDA_PATH)!")
            print("      ⚠️ GPU 가속을 온전히 사용하려면 NVIDIA CUDA Toolkit 설치가 필수입니다.")
            print("      👉 다운로드 링크: https://developer.nvidia.com/cuda-downloads")
            print("      => Reverting to CPU mode to prevent crashes.")
            has_gpu = False
    else:
        print("[1/3] Hardware Scan: ⚪ CPU Mode (No NVIDIA GPU detected)")

    print("[2/3] Checking LLM Engine (llama-cpp-python) Status...")
    engine_status = get_llama_engine_status()
    
    # Pre-built wheel index URLs
    cu121_index = "https://abetlen.github.io/llama-cpp-python/whl/cu121"
    cpu_index = "https://abetlen.github.io/llama-cpp-python/whl/cpu"
    
    if has_gpu and engine_status == "CPU":
        print("=> NVIDIA GPU detected, but CPU engine is installed. Fixing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--extra-index-url", cu121_index, "--force-reinstall", "--no-cache-dir", "--only-binary=llama-cpp-python"])
    elif has_gpu and engine_status == "GPU_MISSING_CUDA":
        print("=> [WARNING] NVIDIA GPU detected, but CUDA 12 Toolkit runtime is missing! Reverting to CPU engine to prevent crash.")
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--extra-index-url", cpu_index, "--force-reinstall", "--no-cache-dir", "--only-binary=llama-cpp-python"])
    elif not has_gpu and engine_status in ["GPU", "GPU_MISSING_CUDA"]:
        print("=> No NVIDIA GPU detected (or no CUDA Toolkit), but GPU engine is installed. Fixing...")
        subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--extra-index-url", cpu_index, "--force-reinstall", "--no-cache-dir", "--only-binary=llama-cpp-python"])
    elif engine_status == "NONE":
        if has_gpu:
            print("=> Engine not found. Installing GPU engine (Pre-built cu121 wheel)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--extra-index-url", cu121_index, "--only-binary=llama-cpp-python"])
        else:
            print("=> Engine not found. Installing CPU engine (Pre-built wheel)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--extra-index-url", cpu_index, "--only-binary=llama-cpp-python"])
    else:
        print("=> Hardware and Engine configuration matches perfectly. 🟢")

    print("\n[3/3] Ensuring basic Nexus dependencies are installed (fastapi, uvicorn, aiohttp)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "aiohttp", "psutil"])

def start_services():
    print("\n🚀 Starting AMEVA Nexus Services...\n")
    processes = []
    
    # 1. Log Server
    p1 = subprocess.Popen([sys.executable, "src/log_server.py"])
    processes.append(("Log Server (Port 14003)", p1))
    
    # 2. Web Dashboard
    p2 = subprocess.Popen([sys.executable, "src/log_web_dashboard.py"])
    processes.append(("Web Dashboard (Port 14001)", p2))
    
    # 3. Model Router (API Gateway)
    time.sleep(2)
    p3 = subprocess.Popen([sys.executable, "src/model_router.py"])
    processes.append(("Model Router API (Port 14000)", p3))
    
    print("\n✅ All services are running!")
    print("   - API Gateway : http://localhost:14000")
    print("   - Dashboard   : http://localhost:14001")
    print("   - Log Server  : http://localhost:14003")
    print("🛑 Press [Ctrl+C] to stop everything cleanly.")
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n🛑 Shutting down all services...")
        for name, p in processes:
            print(f"Terminating {name}...")
            p.terminate()
            try:
                p.wait(timeout=3)
            except subprocess.TimeoutExpired:
                p.kill()
        print("Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    check_and_install_dependencies()
    start_services()
