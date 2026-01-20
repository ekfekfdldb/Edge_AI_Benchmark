# Edge Computing Benchmark

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Jetson_Orin_Nano_%7C_Raspberry_Pi_5-green?style=flat-square)
![Accelerator](https://img.shields.io/badge/Accelerator-NPU_%26_GPU-orange?style=flat-square)

**Raspberry Pi 5 (Hailo-8 NPU)** 와 **NVIDIA Jetson Orin Nano (Embedded GPU)** 환경에서 YOLO 모델의 추론 성능을 측정하고 비교하는 벤치마크 프로그램입니다.

---

##  Project Structure

```text
EDGE COMPUTING BENCHMARKS/
├── common/
│   ├── __init__.py         # 패키지 초기화
│   └── logger.py           # 성능 측정 및 데이터 로깅 모듈
├── logs/                   # 벤치마크 결과(CSV) 자동 저장 경로
├── models/
│   ├── example.engine      # [Jetson] TensorRT 변환 모델
│   └── example.hef         # [RPi 5] Hailo 컴파일 모델
├── videos/
│   └── example.mp4         # 벤치마크 테스트용 영상
├── Jetson.py               # NVIDIA Jetson 실행 스크립트
└── Raspberry_Pi.py         # Raspberry Pi 5 실행 스크립트
```

---

##  Environment & Requirements

각 하드웨어 환경에 맞는 필수 라이브러리를 설치해야 합니다.

### 1. Common (공통)
두 플랫폼 모두 아래 라이브러리가 필요합니다.

```bash
pip install psutil pandas opencv-python
```

### 2. NVIDIA Jetson Orin Nano
* **OS:** Ubuntu 20.04 / 22.04 (JetPack 6.x)
* **Requirements:** YOLOv8 및 Jetson 통계 도구

```bash
pip install jetson-stats ultralytics
```

### 3. Raspberry Pi 5
* **OS:** Raspberry Pi OS (Bookworm 64-bit)
* **Requirements:** Hailo NPU 드라이버 및 Python API

```bash
pip install hailo-platform
```

---

##  Usage

실행 전 `models/` 폴더에 변환된 모델 파일이 있는지 확인하십시오.

### Run on Jetson Orin Nano
```bash
python3 Jetson.py --model models/example.engine --video videos/example.mp4
```

### Run on Raspberry Pi 5
```bash
python3 Raspberry_Pi.py --model models/example.hef --video videos/example.mp4
```

---

## 📊 Output Data (CSV Log)

벤치마크 결과는 `logs/` 폴더에 CSV 파일로 자동 저장됩니다.

| Column Name | Description | Note |
| :--- | :--- | :--- |
| **Frame_ID** | 프레임 순서 | - |
| **Timestamp** | 기록 시간 | - |
| **E2E_Latency_ms** | 전체 지연 시간 | 전처리 + 추론 + 후처리 |
| **Inference_Time_ms** | 연산 시간 | NPU/GPU 순수 추론 시간 |
| **CPU_Usage_%** | CPU 사용률 | 시스템 전체 부하 |
| **Memory_Usage_%** | 메모리 사용률 | RAM 사용량 |
| **Chip_Temp_C** | 칩셋 온도 | Jetson(GPU) / RPi(SoC) |
| **Power_W** | 소비 전력 | Jetson(센서 측정) / RPi(0.0 표기) |