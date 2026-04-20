# src/pc_launcher.py
import subprocess
import sys
import time
import json
import os
from typing import Optional, Iterator

import requests


class AmevaPCNodeLauncher:
    """
    PC에서 llama.cpp server를 Docker로 띄우고,
    같은 객체에서 프롬프트를 보내 결과를 받는 기능까지 제공.
    """

    def __init__(self, config_path: Optional[str] = None, node_key: str = "EXPERT_PC"):
        self.node_key = node_key
        self.config_path = config_path or self._default_config_path()
        self.config = self._load_config()
        self.pc_cfg = self.config["nodes"][self.node_key]
        self.container_name = "ameva_pc_expert"

    @property
    def port(self) -> int:
        return int(self.pc_cfg["port"])

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _default_config_path(self) -> str:
        here = os.path.dirname(os.path.abspath(__file__))
        return os.path.join(here, "..", "config.json")

    def _load_config(self) -> dict:
        if not os.path.exists(self.config_path):
            print(f"[FATAL] config.json not found: {self.config_path}")
            sys.exit(1)
        with open(self.config_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def check_environment(self) -> bool:
        print("[PC-LAUNCHER] Checking Docker availability...")
        try:
            r = subprocess.run(["docker", "info"], capture_output=True, text=True)
            if r.returncode != 0:
                print("[ERROR] Docker daemon not ready. Please start Docker Desktop.")
                print((r.stderr or "").strip()[:400])
                return False
        except FileNotFoundError:
            print("[ERROR] docker command not found. Install Docker Desktop or Docker Engine.")
            return False
        return True

    def check_gpu(self) -> bool:
        try:
            subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            return False

    def is_container_running(self) -> bool:
        """똥컴 기준: 매번 재기동보다 떠 있으면 재사용이 더 빠름"""
        try:
            r = subprocess.run(
                ["docker", "ps", "--filter", f"name={self.container_name}", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            return self.container_name in (r.stdout or "")
        except Exception:
            return False

    def _remove_existing_container(self):
        subprocess.run(["docker", "rm", "-f", self.container_name], capture_output=True)

    def start_docker_llm(self) -> bool:
        port = self.port
        model_dir = self.pc_cfg["model_dir"]
        model_filename = self.pc_cfg["model_filename"]
        ctx_size = int(self.pc_cfg.get("ctx_size", 4096))
        threads = int(self.pc_cfg.get("threads", 8))

        has_gpu = self.check_gpu()

        if has_gpu:
            image = "ghcr.io/ggml-org/llama.cpp:server-cuda"
            gpu_flags = ["--gpus", "all"]
        else:
            image = "ghcr.io/ggml-org/llama.cpp:server"
            gpu_flags = []

        self._remove_existing_container()

        model_dir_abs = os.path.abspath(model_dir)

        docker_cmd = [
            "docker", "run", "-d",
            "--name", self.container_name,
        ] + gpu_flags + [
            "-p", f"{port}:8080",
            "-v", f"{model_dir_abs}:/models",
            image,
            "-m", f"/models/{model_filename}",
            "-c", str(ctx_size),
            "-t", str(threads),
            "--host", "0.0.0.0",
            "--port", "8080"
        ]

        print(f"[PC-LAUNCHER] Starting llama.cpp server container: {image}")
        print(f"[PC-LAUNCHER] HostPort={port}  Model=/models/{model_filename}  ctx={ctx_size}  threads={threads}")

        try:
            subprocess.run(docker_cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"[FATAL] docker run failed: {e}")
            return False

        return True

    def wait_until_ready(self, timeout_sec: int = 60) -> bool:
        deadline = time.time() + timeout_sec
        while time.time() < deadline:
            # 1) /health
            try:
                r = requests.get(f"{self.base_url}/health", timeout=2)
                if r.status_code == 200:
                    return True
            except Exception:
                pass

            # 2) fallback ping /completion
            try:
                r = requests.post(
                    f"{self.base_url}/completion",
                    json={"prompt": "ping", "n_predict": 1, "stream": False},
                    timeout=4
                )
                if r.status_code == 200:
                    return True
            except Exception:
                pass

            time.sleep(1)

        return False

    def launch(self) -> bool:
        if not self.check_environment():
            return False

        # 떠 있으면 재사용
        if self.is_container_running():
            return self.wait_until_ready(timeout_sec=30)

        if not self.start_docker_llm():
            return False
        if not self.wait_until_ready(timeout_sec=60):
            print("[ERROR] llama.cpp server did not become ready in time.")
            return False
        return True

    # =========================================================
    # ✅ 여기부터: "프롬프트 → 도커 → 결과" 기능
    # =========================================================

    def infer_completion(
        self,
        prompt: str,
        n_predict: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        timeout: float = 30.0,
    ) -> str:
        """
        llama.cpp server의 /completion 엔드포인트로 요청을 보내고 텍스트를 반환.
        """
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stream": False
        }

        r = requests.post(f"{self.base_url}/completion", json=payload, timeout=timeout)
        r.raise_for_status()
        data = r.json()

        # llama.cpp 서버 버전에 따라 응답 키가 다를 수 있어 fallback 처리
        if "content" in data and isinstance(data["content"], str):
            return data["content"]
        if "choices" in data and data["choices"]:
            return data["choices"][0].get("text", "")
        if "text" in data:
            return data.get("text", "")

        return ""

    def infer_completion_stream(
        self,
        prompt: str,
        n_predict: int = 512,
        temperature: float = 0.3,
        top_p: float = 0.7,
        top_k: int = 40,
        repeat_penalty: float = 1.1,
        timeout: float = 60.0,
    ) -> Iterator[str]:
        """
        stream=True일 때 chunk를 순차 yield.
        - llama.cpp는 SSE(data: {...}\n\n) 형식일 수 있고,
          또는 line-by-line JSON일 수 있어 둘 다 대응.
        """
        payload = {
            "prompt": prompt,
            "n_predict": n_predict,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repeat_penalty": repeat_penalty,
            "stream": True
        }

        r = requests.post(f"{self.base_url}/completion", json=payload, stream=True, timeout=timeout)
        r.raise_for_status()

        buffer = b""
        for chunk in r.iter_content(chunk_size=1024):
            if not chunk:
                continue
            buffer += chunk

            # SSE 이벤트 경계는 보통 \n\n
            while b"\n\n" in buffer:
                block, buffer = buffer.split(b"\n\n", 1)
                lines = block.decode("utf-8", errors="replace").split("\n")

                for line in lines:
                    line = line.strip()
                    if not line:
                        continue

                    # SSE: data: ...
                    if line.startswith("data:"):
                        payload_str = line[5:].strip()
                        if payload_str == "[DONE]":
                            return
                        try:
                            obj = json.loads(payload_str)
                        except json.JSONDecodeError:
                            continue
                        token = obj.get("content") or obj.get("text") or ""
                        if token:
                            yield token

                    # non-SSE: JSON line
                    else:
                        try:
                            obj = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        token = obj.get("content") or obj.get("text") or ""
                        if token:
                            yield token