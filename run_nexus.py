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
    
    # Since the models are run entirely inside Docker containers (workers),
    # the host machine's virtual environment does not need llama-cpp-python installed.
    # We only need to check and install basic web dependencies.
    print("[1/2] Skipping host-side LLM engine check (Docker handles model execution).")
    print("[2/2] Ensuring basic Nexus dependencies are installed (fastapi, uvicorn, aiohttp, psutil)...")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "fastapi", "uvicorn", "aiohttp", "psutil"])

import socket

def get_local_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"

def check_docker_running():
    try:
        # Run docker info to check if Docker daemon is active
        res = subprocess.run(["docker", "info"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False

def ensure_docker_running():
    print("[Docker] Checking Docker status...")
    if check_docker_running():
        print("   🟢 Docker Daemon : ONLINE (V)")
        return True
        
    print("   ❌ Docker Daemon : OFFLINE (X)")
    
    # Try to launch Docker Desktop on Windows
    docker_desktop_path = r"C:\Program Files\Docker\Docker\Docker Desktop.exe"
    if os.path.exists(docker_desktop_path):
        print("   [+] Attempting to launch Docker Desktop automatically...")
        try:
            os.startfile(docker_desktop_path)
        except Exception as e:
            print(f"   ❌ Failed to launch Docker Desktop automatically: {e}")
    else:
        print("   ⚠️  Docker Desktop executable not found at default path.")
        print("   👉 Please start Docker Desktop manually.")

    # Poll and wait for Docker to start
    print("   ⏱️ Waiting for Docker daemon to become active (timeout 60s)...")
    start_wait = time.time()
    while time.time() - start_wait < 60:
        if check_docker_running():
            print("\n   🟢 Docker Daemon has started! : ONLINE (V)")
            return True
        print(".", end="", flush=True)
        time.sleep(3)
    
    print("\n   ❌ Timeout: Docker daemon did not start.")
    return False

def start_services():
    print("\n🚀 Starting AMEVA Nexus Services...\n")
    
    if not ensure_docker_running():
        print("   ⚠️  [ERROR] Docker is not running! Please start Docker Desktop and run this script again.")
        print("   => Exiting launcher.")
        sys.exit(1)

    print("[Docker] Checking and starting Worker Cluster...")
    try:
        # Start docker-compose without blocking, hiding its noisy output
        subprocess.run(["docker-compose", "up", "-d", "--build"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print("   ✅ Docker Worker Cluster : ONLINE (V)\n")
    except Exception as e:
        print(f"   ❌ Docker Worker Cluster : OFFLINE (X) - Failed to start automatically.")
        print("   => You may need to run `docker-compose up -d --build` manually.\n")

    processes = []
    
    # 1. Log Server
    p1 = subprocess.Popen([sys.executable, "-m", "src.core.logger"])
    processes.append(("Log Server (Port 14003)", p1))
    
    # 2. Web Dashboard
    p2 = subprocess.Popen([sys.executable, "-m", "src.api.dashboard"])
    processes.append(("Web Dashboard (Port 14001)", p2))
    
    # 3. Model Router (API Gateway)
    time.sleep(2)
    p3 = subprocess.Popen([sys.executable, "-m", "src.api.router"])
    processes.append(("Model Router API (Port 14000)", p3))
    
    local_ip = get_local_ip()
    
    # Reset existing workers to OFFLINE in the DB so we don't read stale records
    try:
        from src.core.database import setup_db, get_connection
        setup_db()
        conn = get_connection()
        cursor = conn.cursor()
        cursor.execute("UPDATE router_workers SET status='OFFLINE'")
        conn.commit()
        conn.close()
    except Exception as e:
        pass

    # Wait for both workers to register and be ONLINE
    print("[Launcher] Waiting for GPU and CPU workers to initialize and register...")
    expected_workers = {
        "Docker_8B_GPU": "GPU Server (Llama-8)",
        "Docker_3B_CPU": "CPU Server (qwen-3)"
    }
    ready_workers = set()
    start_time = time.time()
    
    try:
        from src.core.database import DatabaseManager
        while len(ready_workers) < len(expected_workers):
            # Check if router process exited early
            if p3.poll() is not None:
                print("   ❌ Model Router API exited unexpectedly!")
                break
                
            workers = DatabaseManager.router_get_workers()
            for w in workers:
                name = w['worker_name']
                status = w['status']
                if name in expected_workers and status == 'ONLINE':
                    if name not in ready_workers:
                        ready_workers.add(name)
                        print(f"   ✅ {expected_workers[name]} : ONLINE (V)")
                        
            if len(ready_workers) < len(expected_workers):
                elapsed = int(time.time() - start_time)
                # Print loading status
                print(f"   ⏱️ Waiting for workers... (Online: {len(ready_workers)}/{len(expected_workers)}) [{elapsed}s]", end="\r", flush=True)
                time.sleep(2)
        print("\n")
    except Exception as e:
        print(f"\n⚠️ Error waiting for workers: {e}")

    print("\n✅ All services are running and workers are ONLINE!")
    print(f"   - API Gateway : http://{local_ip}:14000 (External) | http://localhost:14000 (Local)")
    print(f"   - Dashboard   : http://{local_ip}:14001 (External) | http://localhost:14001 (Local)")
    print(f"   - Log Server  : http://{local_ip}:14003 (External) | http://localhost:14003 (Local)")
    
    print("\n=========================================================================")
    print(" 📖 [Quick API Usage Guide] - Copy & Paste to your team!")
    print("=========================================================================")
    print(f" [1] Check API Docs & Available Models:")
    print(f"     curl http://{local_ip}:14000/help")
    print("")
    print(f" [2] Send a Chat Request (Streaming Mode):")
    print(f"     curl -X POST http://{local_ip}:14000/api/chat \\")
    print("          -H \"Content-Type: application/json\" \\")
    print("          -d '{\"model\": \"Llama-8\", \"prompt\": \"Hello!\", \"stream\": true}'")
    print("=========================================================================\n")
    
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
        
        print("\n🐳 Releasing Docker resources (stopping containers)...")
        try:
            subprocess.run(["docker-compose", "down"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("   ✅ Docker containers stopped and removed.")
        except Exception as e:
            print(f"   ❌ Failed to stop Docker containers: {e}")
            
        print("Goodbye!")
        sys.exit(0)

if __name__ == "__main__":
    check_and_install_dependencies()
    start_services()
