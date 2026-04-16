# AMEVA Model Nexus (Centralized LLM Serving Hub)

AMEVA Model Nexus는 여러 프로젝트와 디바이스(Desktop, Edge ARM, Mobile) 환경에서
개별적으로 관리되던 대용량 GGUF 기반 LLM 모델들을 **단일 중앙 서버로 통합**하기 위해 설계된
로컬 LLM 서빙 허브입니다.

이 시스템의 목적은 단순한 모델 호스팅이 아니라,
**제한된 하드웨어 자원을 안정적으로 분배하면서도 병렬 추론 성능을 최대한 끌어내는 것**입니다.

---

## Problem Statement

로컬 LLM 환경에서 공통적으로 발생하는 문제는 다음과 같습니다.

- 여러 프로젝트가 동일한 모델을 각각 보관하여 발생하는 스토리지 중복
- CPU/GPU 자원 경쟁으로 인한 OOM, 발열, 성능 급락
- 병렬 요청 증가 시 예측 불가능한 latency 폭증
- 추론 중 장애가 발생할 경우 서버 전체가 멈추는 현상

AMEVA Model Nexus는 이러한 문제를 **소프트웨어 아키텍처 차원에서 제어**하는 것을 목표로 합니다.

---

## Design Goals

1. **Centralization**
   - 모델 파일(GGUF)을 단일 고성능 스토리지에 집중 관리
   - 여러 애플리케이션이 네트워크를 통해 동일한 모델을 공유

2. **Resource-Aware Scheduling**
   - 단순 QPS 제한이 아닌, 실시간 CPU/RAM 상태 기반의 추론 허용 제어
   - 하드웨어 한계에 근접하는 상황에서도 서버가 응답성을 유지하도록 설계

3. **Operational Safety**
   - 추론 지연, 무한 대기, 메모리 포화 상황에서 서버 전체 장애로 전파되지 않도록 차단
   - 관측 가능성(Observability)을 1차 설계 요소로 포함

---

## Architecture Overview

AMEVA Model Nexus는 FastAPI 기반의 단일 프로세스 서버로 구성되며,
다음과 같은 핵심 구성 요소를 포함합니다.

### 1. Centralized Model Runtime

- llama.cpp 기반 GGUF 모델을 서버 시작 시 또는 런타임에 로드
- 모든 추론 요청은 중앙 모델 인스턴스를 통해 처리
- 각 클라이언트는 로컬에 모델을 보관할 필요 없음

### 2. Chomchomsky Smart Throttling

- psutil 기반 CPU/RAM 사용량을 1초 주기로 중앙에서 수집
- 요청마다 자원을 직접 측정하지 않고, **캐시된 전역 상태**를 기준으로 입장 여부 판단
- 최대 병렬 추론 수(MAX_CONCURRENT_CHUNKS)를 상한선으로 두되,
  실제 허용 여부는 시스템 여유 상태에 따라 동적으로 결정

이 설계는 다음과 같은 트레이드오프를 가집니다.

- 장점: 자원 포화 시에도 서버가 즉시 응답을 반환할 수 있음
- 단점: peak throughput은 환경에 따라 제한될 수 있음

### 3. Concurrency Control and Timeout Guard

- asyncio.Semaphore를 통한 명시적 병렬 수 제한
- ThreadPoolExecutor를 사용하여 C++ 기반 추론 연산을 이벤트 루프와 분리
- `asyncio.wait_for` 기반의 추론 타임아웃 적용
  - 개별 요청의 지연이 전체 서버 안정성을 해치지 않도록 방지

### 4. Elastic Logging and Log Rotation

- 요청 경로에서 디스크 I/O를 발생시키지 않도록 로그를 메모리 버퍼에 적재
- 백그라운드 워커가 주기적으로 로그를 파일로 flush
- 파일 크기 기준(100MB) 로테이션 및 gzip 압축 적용

이 방식은 로그 신뢰성과 런타임 성능 사이의 균형을 고려한 선택입니다.

---

## Observability and Operations

### Observatory API (`/stats`)

서버는 실시간 운영 상태를 JSON 형태로 제공합니다.

- 현재 처리 중인 추론 요청 수
- 완료된 요청 수 및 평균 추론 시간
- 최근 60초간의 CPU/RAM 사용 히스토리

이를 기반으로 외부 대시보드(Grafana 등)와의 연동을 전제로 설계되었습니다.

---

## Trade-offs and Non-Goals

- GPU 기반 대규모 병렬 추론은 의도적으로 범위에서 제외
  - VRAM 공유 및 컨텍스트 충돌 리스크를 명확히 인지한 선택
- 단일 서버 내 안정성을 우선시하며,
  분산 클러스터링은 상위 레이어에서 해결하는 것을 가정

---

## Intended Use Cases

- 사내 문서 요약, 검색, 분석을 위한 내부 LLM 허브
- 여러 애플리케이션이 동시에 접근하는 중앙 AI 서비스
- 자원 제약 환경에서의 안정적인 로컬 LLM 운영 실험

---

## Summary

AMEVA Model Nexus는
“최대 성능”보다는 “통제 가능한 성능”을 목표로 설계된
**자원 인식형 로컬 LLM 서빙 아키텍처**입니다.

이 프로젝트는 모델 자체보다,
**모델을 어떻게 안전하게 운영할 것인가에 대한 고민의 결과물**입니다.