# Edge Computing Benchmark

![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)
![Platform](https://img.shields.io/badge/Platform-Jetson_Orin_Nano_%7C_Raspberry_Pi_5-green?style=flat-square)
![Accelerator](https://img.shields.io/badge/Accelerator-NPU_%26_GPU-orange?style=flat-square)

**Raspberry Pi 5 (Hailo-8 NPU)** 와 **NVIDIA Jetson Orin Nano (Embedded GPU)** 환경에서 YOLO 모델의 추론 성능을 측정하고 비교하는 벤치마크 프로그램입니다.



##  Project Structure

```text
EDGE COMPUTING BENCHMARKS/
├── common/
│   ├── __init__.py         # 패키지 초기화
│   └── logger.py           # 성능 측정 및 데이터 로깅 모듈
├── logs/                   # [Output] 벤치마크 결과(CSV) 자동 저장 경로
├── output_videos/          # [Output] 시각화 결과 영상(.mp4) 저장 경로
├── model/
│   ├── yolop.engine        # [Jetson] TensorRT 변환 모델
│   └── yolop.hef           # [RPi 5] Hailo 컴파일 모델
├── videos/
│   └── test.mp4            # 벤치마크 테스트용 원본 영상
├── Jetson.py               # [Benchmark] Jetson 성능 측정용 (TensorRT 기반)
├── Jetson_visualize.py     # [Visualization] Jetson 검증 및 영상 저장용
├── Raspberry_Pi.py         # [Benchmark] Raspberry Pi 5 성능 측정용 (Hailo 기반)
└── Raspberry_Pi_visualize.py # [Visualization] RPi 5 검증 및 영상 저장용
```



##  Environment & Requirements

각 하드웨어 환경에 맞는 필수 라이브러리를 설치해야 합니다.

1. Common (공통)
두 플랫폼 모두 아래 라이브러리가 필요합니다.

```Bash
pip install psutil opencv-python
```

2. NVIDIA Jetson Orin Nano
OS: Ubuntu 20.04 / 22.04 (JetPack 6.x)

Pre-requisites: JetPack SDK가 설치되어 있어야 합니다.

```Bash
# 1. 시스템 업데이트 및 pip 설치
sudo apt-get update && sudo apt-get install -y python3-pip

# 2. PyCUDA 환경변수 설정 및 설치
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
pip3 install pycuda --user

# 3. 필수 라이브러리 설치
pip3 install psutil opencv-python --user
```

3. Raspberry Pi 5
OS: Raspberry Pi OS (Bookworm 64-bit)

Pre-requisites: Hailo 하드웨어가 연결되어 있어야 합니다.



Step 1: 시스템 드라이버 설치 (최초 1회)

```Bash
sudo apt update && sudo apt install -y hailo-all
sudo reboot
```

Step 2: 가상 환경 설정 및 라이브러리 설치 (재부팅 후 실행)

```Bash

# 가상 환경 생성 및 활성화
python3 -m venv hailo_env
source hailo_env/bin/activate

# 라이브러리 설치
pip install hailo-platform psutil opencv-python
```


##  Usage

실행 전 `models/` 폴더에 변환된 모델 파일이 있는지 확인하십시오.

**Run on Jetson Orin Nano**

성능 측정 (Benchmark):

```Bash
python3 Jetson.py --model model/yolop.engine --video videos/test.mp4
```
결과 영상 저장 (Visualization):

```Bash
python3 Jetson_visualize.py --model model/yolop.engine --video videos/test.mp4
```
**Run on Raspberry Pi 5**

(반드시 가상 환경 활성화 후 실행: source hailo_env/bin/activate)

성능 측정 (Benchmark):

```Bash
python3 Raspberry_Pi.py --model model/yolop.hef --video videos/test.mp4
```
결과 영상 저장 (Visualization):

```Bash
python3 Raspberry_Pi_visualize.py --model model/yolop.hef --video videos/test.mp4
```

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