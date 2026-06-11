import os
import sys
import subprocess
import time

def check_nvidia_gpu():
    try:
        # Check if nvidia-smi exists and executes correctly
        subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def check_cuda_toolkit():
    try:
        # Check if nvcc (CUDA compiler) is available
        subprocess.run(["nvcc", "--version"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def check_and_install_dependencies():
    has_gpu = check_nvidia_gpu()
    has_cuda = check_cuda_toolkit()
    
    print("=======================================")
    print(" 🚀 AMEVA Nexus Smart Launcher")
    print("=======================================")
    
    if has_gpu and has_cuda:
        print("[1/3] Hardware Scan: 🟢 NVIDIA GPU & CUDA Toolkit Found! (Full GPU Mode)")
    elif has_gpu and not has_cuda:
        print("[1/3] Hardware Scan: 🟡 NVIDIA GPU Found, BUT missing CUDA Toolkit!")
        print("      ⚠️ GPU 가속을 온전히 사용하려면 NVIDIA CUDA Toolkit 설치가 필수입니다.")
        print("      👉 다운로드 링크: https://developer.nvidia.com/cuda-downloads")
        print("      (지금은 임시로 CPU 모드 또는 제한된 GPU 모드로 작동합니다.)")
    else:
        print("[1/3] Hardware Scan: ⚪ CPU Mode (No NVIDIA GPU detected)")
    
    # Check PyTorch installation
    try:
        import torch
        torch_installed = True
        torch_cuda = torch.cuda.is_available()
    except ImportError:
        torch_installed = False
        torch_cuda = False

    # Install / Update logic
    if has_gpu and has_cuda:
        if not torch_installed or not torch_cuda:
            print("[2/3] Installing PyTorch with CUDA support...")
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio", "--index-url", "https://download.pytorch.org/whl/cu118"])
        else:
            print("[2/3] PyTorch (CUDA) is already installed.")
            
        # Check llama-cpp-python
        try:
            import llama_cpp
            print("[3/3] llama-cpp-python is already installed. (Assuming CUDA build)")
        except ImportError:
            print("[3/3] Installing llama-cpp-python with CUDA support...")
            env = os.environ.copy()
            env["CMAKE_ARGS"] = "-DGGML_CUDA=on"
            try:
                subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python", "--upgrade", "--force-reinstall", "--no-cache-dir"], env=env, check=True)
            except subprocess.CalledProcessError:
                print("⚠️ Failed to compile CUDA version of llama-cpp-python. (Missing C++ Build Tools?)")
                print("⚠️ Falling back to pre-built CPU version...")
                subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
    else:
        # CPU Mode (or GPU without CUDA Toolkit)
        if not torch_installed:
            print("[2/3] Installing PyTorch (CPU)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "torch", "torchvision", "torchaudio"])
        else:
            print("[2/3] PyTorch (CPU) is already installed.")
            
        try:
            import llama_cpp
            print("[3/3] llama-cpp-python is already installed.")
        except ImportError:
            print("[3/3] Installing llama-cpp-python (CPU)...")
            subprocess.run([sys.executable, "-m", "pip", "install", "llama-cpp-python"])
            
    # Ensure basic dependencies for the Nexus system
    print("\n[+] Ensuring basic dependencies are installed (fastapi, uvicorn, aiohttp)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "aiohttp", "psutil"])

def start_services():
    print("\n🚀 Starting AMEVA Nexus Services...\n")
    processes = []
    
    # 1. Log Server
    p1 = subprocess.Popen([sys.executable, "src/log_server.py"])
    processes.append(("Log Server (Port 9999)", p1))
    
    # 2. Web Dashboard
    p2 = subprocess.Popen([sys.executable, "src/log_web_dashboard.py"])
    processes.append(("Web Dashboard (Port 14000)", p2))
    
    # 3. Model Router (Simulation)
    time.sleep(2) # Give servers a moment to start
    p3 = subprocess.Popen([sys.executable, "src/model_router.py"])
    processes.append(("Model Router (Watchdog & Dispatcher)", p3))
    
    print("\n✅ All services are running! Web UI available at: http://localhost:14000")
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
