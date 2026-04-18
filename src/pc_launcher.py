import subprocess
import sys
import time
import json
import os

class AmevaPCNodeLauncher:
    def __init__(self, config_file="config.json"):
        """런처 초기화 및 설정 파일 로드"""
        self.config_file = config_file
        self.config = self._load_config()
        self.pc_cfg = self.config["nodes"]["EXPERT_PC"]

    def _load_config(self):
        """내부 메서드: 설정 파일을 읽어옵니다."""
        if not os.path.exists(self.config_file):
            print(f"❌ {self.config_file} 파일을 찾을 수 없습니다!")
            sys.exit(1)
        with open(self.config_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def check_environment(self) -> bool:
        """WSL 및 Docker 구동 상태를 점검합니다."""
        print("🔍 [AMEVA PC Node] 시스템 환경 점검...")
        time.sleep(0.5)

        # 1. WSL 점검
        try:
            if subprocess.run(["wsl", "-l", "-v"], capture_output=True).returncode != 0:
                print("❌ WSL(Linux 하위 시스템)이 꺼져있네요 ㅠ")
                return False
        except FileNotFoundError:
            print("❌ 시스템에 WSL 명령어가 없네요 ㅠ")
            return False

        # 2. Docker 점검
        try:
            if subprocess.run(["docker", "info"], capture_output=True).returncode != 0:
                print("❌ 도커(Docker Desktop)가 꺼져있네요 ㅠ 백그라운드 고래를 확인해주세요!")
                return False
        except FileNotFoundError:
            print("❌ 도커가 설치되지 않았네요 ㅠ")
            return False
            
        return True

    def check_gpu(self) -> bool:
        """NVIDIA GPU 존재 여부를 확인합니다."""
        try:
            subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            print("🔥 NVIDIA GPU가 감지되었습니다. [CUDA 가속 모드]로 기동합니다.")
            return True
        except (FileNotFoundError, subprocess.CalledProcessError):
            print("❄️ NVIDIA GPU가 감지되지 않았습니다. [CPU 전용 모드]로 우회 기동합니다.")
            return False

    def start_docker_llm(self):
        """Docker 컨테이너를 실행하여 AI 모델을 띄웁니다."""
        port = self.pc_cfg["port"]
        model_dir = self.pc_cfg["model_dir"]
        model_filename = self.pc_cfg["model_filename"]
        ctx_size = self.pc_cfg["ctx_size"]

        print(f"\n🚀 AI 노드 점화 준비! (모델: {model_filename}, 포트: {port})")
        
        # 중복 컨테이너 정리
        subprocess.run(["docker", "rm", "-f", "ameva_pc_expert"], capture_output=True)

        # GPU 유무에 따른 분기
        has_gpu = self.check_gpu()
        
        if has_gpu:
            docker_image = "ghcr.io/ggerganov/llama.cpp:server-cuda"
            gpu_flags = ["--gpus", "all"]
        else:
            docker_image = "ghcr.io/ggerganov/llama.cpp:server"
            gpu_flags = []

        # 도커 실행 커맨드 조립
        docker_cmd = [
            "docker", "run", "-d",
            "--name", "ameva_pc_expert",
        ] + gpu_flags + [
            "-p", f"{port}:8080",
            "-v", f"{model_dir}:/models",
            docker_image,
            "-m", f"/models/{model_filename}",
            "-c", str(ctx_size),
            "--host", "0.0.0.0",
            "--port", "8080"
        ]

        try:
            subprocess.run(docker_cmd, check=True)
            mode_text = "GPU(CUDA)" if has_gpu else "CPU Only"
            print(f"\n🎉 성공! AMEVA PC 노드가 도커 위에서 실행 중입니다. [{mode_text}]")
            print(f"👉 API 주소: http://localhost:{port}/v1/chat/completions")
        except subprocess.CalledProcessError as e:
            print(f"❌ 도커 컨테이너 실행 중 에러: {e}")
            sys.exit(1)

    def launch(self):
        """환경 점검 후 컨테이너를 최종 실행하는 메인 시퀀스"""
        if self.check_environment():
            self.start_docker_llm()
        else:
            print("\n⛔ 환경 점검을 통과하지 못해 AI 노드 점화를 취소합니다.")
            sys.exit(1)