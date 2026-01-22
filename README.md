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
├── logs/                   # [Output] 벤치마크 결과(CSV) 자동 저장 경로
├── output_videos/          # [Output] 시각화 결과 영상(.mp4) 저장 경로
├── models/
│   ├── yolop.engine        # [Jetson] TensorRT 변환 모델
│   └── yolop.hef           # [RPi 5] Hailo 컴파일 모델
├── videos/
│   └── test.mp4            # 벤치마크 테스트용 원본 영상
├── Jetson_Universal.py     # [Benchmark] Jetson 성능 측정용 (TensorRT 기반)
├── Jetson_Vis.py           # [Visualization] 결과 검증 및 영상 저장용
└── Raspberry_Pi.py         # [Benchmark] Raspberry Pi 5 실행 스크립트
```

---

##  Environment & Requirements

각 하드웨어 환경에 맞는 필수 라이브러리를 설치해야 합니다.

### 1. Common
두 플랫폼 모두 아래 라이브러리가 필요합니다.

```bash
pip install psutil pandas opencv-python
```

### 2. NVIDIA Jetson Orin Nano
* **OS:** Ubuntu 20.04 / 22.04 (JetPack 6.x)
* **Requirements:** YOLOv8 및 Jetson 통계 도구

```bash
pip install jetson-stats ultralytics
pip install jetson-stats pycuda #TensorRT
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
python3 -u Jetson_Universal.py --model models/yolop.engine --video videos/test.mp4 #TensorRT
python3 Jetson_Vis.py --model models/yolop.engine --video videos/test.mp4 #검증 영상
```

### Run on Raspberry Pi 5
```bash
python3 Raspberry_Pi.py --model models/example.hef --video videos/example.mp4
```

---

## Output Data (CSV Log)

벤치마크 결과는 `logs/` 폴더에 CSV 파일로 자동 저장됩니다.

| Column Name | Description | Note |
| :--- | :--- | :--- |
| **Frame_ID** | 프레임 번호 | - |
| **Timestamp** | 기록 시간 | ms 단위 포함 |
| **FPS** | 초당 프레임 수 | 성능의 핵심 지표 (높을수록 좋음) |
| **E2E_Latency_ms** | 전체 지연 시간 | (낮을수록 좋음) |
| **Inference_Time_ms** | 순수 AI 연산 시간 | (낮을수록 좋음) |
| **CPU_Freq_MHz** | CPU 클럭 속도 | 쓰로틀링(성능저하) 감지용 |
| **CPU_Usage_%** | CPU 사용률 | - |
| **GPU_Usage_%** | GPU 사용률 | - |
| **Memory_Usage_%** | RAM 사용률 | - |
| **Temperature_C** | 칩셋 온도 | - |