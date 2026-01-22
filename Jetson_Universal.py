import cv2
import time
import argparse
import sys
import os
import numpy as np
import csv
from datetime import datetime
import psutil  

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    print("\n[CRITICAL] PyCUDA 또는 TensorRT가 설치되지 않았습니다.")
    sys.exit(1)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

def get_system_temp():
    zones = [
        "/sys/class/thermal/thermal_zone0/temp", 
        "/sys/class/thermal/thermal_zone1/temp", 
        "/sys/class/thermal/thermal_zone2/temp"
    ]
    for zone in zones:
        if os.path.exists(zone):
            try:
                with open(zone, "r") as f:
                    return int(f.read().strip()) / 1000.0
            except:
                continue
    return -1

class StandaloneLogger:
    def __init__(self, model_path, video_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        if not os.path.exists("logs"):
            os.makedirs("logs")
            
        self.filename = f"logs/{timestamp}_Jetson_{model_name}.csv"
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)
        
        self.writer.writerow([
            'Frame_ID', 'Timestamp', 'Latency_ms', 'CPU_Usage_%', 
            'Memory_Usage_%', 'Temperature_C', 'Model', 'Video'
        ])
        print(f"[INFO] 로그 파일 생성됨: {self.filename}")

    def log(self, frame_id, latency, temp, model, video):
        # [수정됨] 추가 데이터 수집
        now_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cpu_usage = psutil.cpu_percent(interval=None)
        mem_usage = psutil.virtual_memory().percent
        
        self.writer.writerow([
            frame_id, now_time, latency, cpu_usage, 
            mem_usage, temp, model, video
        ])
        self.file.flush()

    def close(self):
        self.file.close()

class TRTWrapper:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        
        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"[CRITICAL] 엔진 로드 에러: {e}")
            sys.exit(1)
        
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        self.inputs = []
        self.outputs = []
        self.allocations = []
        
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = self.engine.get_tensor_shape(name)
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            size = abs(trt.volume(shape))
            
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            self.allocations.append((host_mem, device_mem))
            self.context.set_tensor_address(name, int(device_mem))

            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()


def run_experiment(model_path, video_path):
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        sys.exit(1)

    print(f"[INFO] 엔진 로딩 중... {model_path}")
    trt_wrapper = TRTWrapper(model_path)
    
    cap = cv2.VideoCapture(video_path)
    logger = StandaloneLogger(model_path, video_path)

    print("[SYSTEM] 온도 센서 확인 중...", end="", flush=True)
    for _ in range(30):
        if get_system_temp() > 0:
            print(" [OK]")
            break
        time.sleep(1)
        print(".", end="", flush=True)

    print(f"[INFO] >>> 실험 시작 (종료: Ctrl+C) <<<")
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            resized = cv2.resize(frame, (640, 384))
            input_data = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
            input_data = np.expand_dims(input_data, axis=0)

            t0 = time.perf_counter_ns()
            trt_wrapper.infer(input_data)
            t1 = time.perf_counter_ns()
            
            latency = (t1 - t0) / 1_000_000.0
            temp = get_system_temp()

            logger.log(frame_id, latency, temp, model_path, video_path)

            if frame_id % 100 == 0:
                print(f"[{frame_id} Frame] Latency: {latency:.2f}ms | Temp: {temp}C", flush=True)
            frame_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단.")
    finally:
        cap.release()
        logger.close()
        print("[INFO] 저장 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    args = parser.parse_args()
    
    run_experiment(args.model, args.video)