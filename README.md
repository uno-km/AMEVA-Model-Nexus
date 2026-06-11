# 📊 AMEVA Model Nexus: Zero-Downtime Distributed LLM Gateway

## 1. 개요 (Abstract)
**AMEVA Model Nexus**는 제한된 물리적 인프라 환경에서 대규모 언어 모델(LLM)을 안정적이고 효율적으로 서빙하기 위해 설계된 엔터프라이즈급 API 게이트웨이 및 분산 워커 클러스터 시스템입니다. 단일 노드의 리소스 병목을 해결하기 위해 워커(Worker)와 라우터(Router) 간의 느슨한 결합(Loosely Coupled) 아키텍처를 채택하였으며, 큐(Queue) 버퍼링 기반의 Zero-Downtime 핫스왑 기능을 통해 무중단 모델 배포 환경을 제공합니다. 

이 프로젝트는 무거운 프레임워크나 불필요한 추상화를 배제하고, Raw Python과 FastAPI를 활용하여 데이터 무결성을 보장하며 극한의 메모리 최적화를 달성합니다.

---

## 2. 주요 기술적 특징 (Technical Deep-Dive)

### 2.1. 데이터 획득 및 전처리 (Data Engineering & processing)
- **비동기 스트리밍 파이프라인**: Server-Sent Events(SSE) 규격을 준수하는 비동기 이벤트 제너레이터를 통해 텍스트 생성 결과를 토큰 단위로 실시간 클라이언트에게 전송합니다.
- **상태 기반 큐(Queue) 버퍼링**: 워커가 다운되거나 모델을 교체(Hot-Swap)하는 순간 발생하는 모든 Inbound 요청을 `PENDING` 상태로 큐에 보관하여 연결 유실률 0%를 달성합니다.

### 2.2. 모델 아키텍처 및 학습 전략 (Model Architecture & Strategy)
- **GGUF 포맷 기반 엣지 인퍼런스**: `llama-cpp-python`을 코어 엔진으로 사용하여 VRAM이 부족한 상황에서도 CPU RAM으로 오버플로우 시키는 하이브리드 추론을 지원합니다.
- **Hardware-Aware 디스패칭**: 윈도우 호스트 환경의 CUDA 유무를 능동 스캔하고, GPU Passthrough가 설정된 워커와 CPU-only 워커를 논리적으로 분리하여 로드밸런싱합니다.

### 2.3. 양자화 및 배포 최적화 (Optimization & Quantization)
- **동적 가중치 스와핑**: `Q4_K_M` 수준으로 양자화된 GGUF 모델들을 도커 볼륨 마운트로 매핑하고, 핫스왑 명령 시 Python의 `gc.collect()`를 명시적으로 호출해 VRAM 단편화를 방지하고 모델을 즉시 스와핑합니다.

### 2.4. 핵심 소스코드 및 실주소 명세 (Core Code Snippets)
#### 2.4.1. 비동기 큐 기반 무중단 핫스왑(Hot-Swap) 로직
* **물리적 소스코드 주소**: [src/api/router.py](file:///C:/Users/GAME/Desktop/uno-km/dev/AMEVA-Model-Nexus/src/api/router.py#L220-L235)
```python
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
```

---

## 3. 시스템 아키텍처 설계 (Software Architecture Design)
```mermaid
graph TD
    subgraph "Client Layer"
        A[Mobile/PC Apps] -- "POST /api/chat" --> B(API Gateway)
        A2[Admin User] -- "POST /admin/hotswap" --> B
    end

    subgraph "Nexus Router Layer (Port 14000)"
        B -- "Assign Task" --> C[(SQLite Queue DB)]
        B -- "Push Log" --> L(Log Ingester)
        W[Watchdog] -- "Monitor / Timeout" --> C
    end

    subgraph "Worker Layer (Docker Cluster)"
        D["worker_8b_gpu (Llama)"] -- "Poll Task & Stream" --> B
        E["worker_3b_cpu (Qwen)"] -- "Poll Task & Stream" --> B
    end

    subgraph "Persistence & Monitoring"
        C -- "Read Logs" --> F[Log Dashboard (Port 14001)]
        L -- "WAL Mode Insert" --> G[(Universal Logs DB)]
    end
```

### 디렉토리 구조 (Repository Layout)
```text
AMEVA-Model-Nexus/
├── run.bat                     # Windows CLI 진입점
├── run_nexus.py                # 시스템 오케스트레이션 및 의존성 주입
├── docker-compose.yml          # 무중단 워커 클러스터 배포 (8B/3B)
├── Dockerfile.worker           # LLM 워커 노드 이미지
├── README.md                   # 프리미엄 기술 명세서
└── src/
    ├── api/                    # 클라이언트 접점 (Gateway & UI)
    │   ├── router.py           # API Gateway & 핫스왑 제어
    │   └── dashboard.py        # 관제 대시보드
    ├── core/                   # 영구 저장소 및 중앙 시스템
    │   ├── database.py         # SQLite 큐/라우팅 스키마 매니저
    │   └── logger.py           # 중앙 로깅 서버 (WAL 모드)
    └── nodes/                  # 엣지 디바이스/컨테이너 실행단
        └── worker.py           # LLM 추론 엔진 및 넥서스 연동 에이전트
```

---

## 4. 데이터 무결성 및 설명성 감사 체계 (Data Integrity & Quality Audit)
- **WAL(Write-Ahead Logging) 모드 DB**: `universal_logs.db`는 초당 수천 건의 로그와 핫스왑 이벤트를 누락 없이 기록하기 위해 SQLite의 WAL 모드를 강제 활성화하고 10초 주기의 배치 청크 인서트를 수행합니다.
- **스트리밍 추적성**: 스트리밍이 완료되거나 타임아웃 오류가 발생하면, 모든 파편화된 토큰을 조합한 `final_result`와 처리 시간을 로거로 전달해 영구 보존 아티팩트로 남깁니다.

---

## 5. 설치 및 파이프라인 가이드 (Execution Pipeline)

> [!IMPORTANT]  
> 이 프로젝트는 윈도우 환경에 최적화되어 있으며, GPU 워커 가동을 위해 NVIDIA 그래픽 드라이버 및 Docker Desktop 설치가 권장됩니다.

1. **Nexus API Gateway 구동**
   ```powershell
   .\run.bat
   ```
   > 런처가 하드웨어를 스캔하여 CUDA 툴킷 부재를 자동 감지하고 Fallback 모드를 구성합니다.

2. **Docker Worker Cluster 구동**
   ```bash
   docker-compose up -d --build
   ```
   > 호스트의 `C:\ameva\models\llm` 경로를 볼륨 마운트하여 대규모 모델을 낭비 없이 로드합니다.

3. **제로 다운타임 핫스왑(Hot-Swap) 실행**
   ```bash
   curl -X POST http://localhost:14000/admin/hotswap \
        -H "Content-Type: application/json" \
        -d '{"target_worker": "Docker_8B_GPU", "new_model_path": "/models/Qwen2.5-7B.gguf", "new_alias": "Qwen-7"}'
   ```

---

## 6. 실험 로드맵 및 검증 전략 (Experimental Roadmap)

| Phase | 목표치 | 검증 모델 | 주요 적용 기법 | 벤치마크 목적 함수 | 상태 |
|-------|--------|-----------|---------------|-------------------|------|
| **Phase 1** | 단일 노드 테스트 | Llama-3-8B | CPU/GPU Fallback | Response Time < 1000ms | 완료 |
| **Phase 2** | 무중단 핫스왑 | Llama-8 -> Qwen-7 | 큐 버퍼링 대기열 | Connection Drop Rate = 0% | 완료 |
| **Phase 3** | 멀티 워커 분산 | Qwen-3, Llama-8 | Round-Robin 분산 | $\max(\text{Throughput})$ | 진행 중 |

---

## 7. 아키텍처 설계 철학 및 트레이드오프 (Architecture Philosophy)

- **Headless 중심의 로컬라이징(Localizing)**: 복잡한 React/Vue GUI를 배제하고 순수 마크업 기반의 초경량 대시보드와 파이썬 CLI를 도입. OS 종속성 충돌을 원천 차단하고 서버 자원의 100%를 LLM 추론에 할당.
- **안정적인 구동(Stable)**: 워커가 크래시나도 Watchdog이 즉시 감지하여 PENDING 큐로 작업을 되돌려 복구(Self-Healing).

| 결정 사항 (Changes) | 이유 (Reason) | 장점 (Pros) | 단점 (Cons) | 획득 이익 (Benefits) |
|--------------------|--------------|------------|------------|----------------------|
| **도커 블루/그린 배포 포기** | VRAM 한계 | VRAM을 단일 모델에 온전히 집중 | 교체 시 약간의 큐 지연 | 제한된 장비에서 거대 모델 무중단 교체 달성 |
| **FastAPI + SQLite** | 컴포넌트 경량화 | Redis, RabbitMQ 설치 불필요 | 초거대 트래픽에는 불리함 | 제로 세팅(Zero-Config) 원클릭 배포 가능 |

---

## 8. 👨‍💻 Tech Stack
- **UI Architecture**: Vanilla HTML/CSS, Server-Sent Events (SSE)
- **Infrastructure**: Docker Compose, Windows PowerShell Scripting
- **Inference**: llama-cpp-python, GGUF, CUDA 12.1 Passthrough
- **Engine Core**: FastAPI, Uvicorn, Asyncio Queueing
- **Backend**: Python 3.11+, SQLite (WAL Mode), Pydantic

---

> **Contact**: AMEVA Engineering Team
> **AMEVA v2.1 "Nexus"** - *Zero-downtime distributed routing for edge AI.*

---
> **"데이터가 장인정신을 만나면, 인공지능은 예술이 된다."** - AMEVA Project