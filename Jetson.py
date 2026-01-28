import cv2
import time
import argparse
import sys
import os
import numpy as np
import csv
import threading
import queue
import subprocess
import re
from datetime import datetime

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule

# =============================
# 1. CUDA 전처리 커널
# =============================
cuda_preprocess = SourceModule(r'''
__global__ void preprocess(unsigned char* input, float* output, int in_w, int in_h, int out_w, int out_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;
    int src_x = x * in_w / out_w;
    int src_y = y * in_h / out_h;
    int in_idx = (src_y * in_w + src_x) * 3;
    int out_plane = out_w * out_h;
    output[y * out_w + x] = input[in_idx + 2] / 255.0f;
    output[out_plane + y * out_w + x] = input[in_idx + 1] / 255.0f;
    output[2 * out_plane + y * out_w + x] = input[in_idx + 0] / 255.0f;
}
''')
preprocess_kernel = cuda_preprocess.get_function("preprocess")

# =============================
# 2. 시스템 모니터링 (Tegrastats 파싱)
# =============================
class SystemMonitor(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.running = True
        self.stats = {"cpu": 0, "gpu": 0, "temp": 0, "power": 0}
        self.daemon = True

    def run(self):
        # tegrastats 실행 (500ms 단위)
        process = subprocess.Popen(['tegrastats', '--interval', str(int(self.interval*1000))], 
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        while self.running:
            line = process.stdout.readline()
            if line:
                # GPU 사용량 추출
                gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
                if gpu_match: self.stats["gpu"] = int(gpu_match.group(1))
                
                # CPU 사용량 추출 (평균값)
                cpu_match = re.findall(r'(\d+)%@', line)
                if cpu_match: self.stats["cpu"] = sum(map(int, cpu_match)) / len(cpu_match)
                
                # 온도 추출 (AO 온도 기준)
                temp_match = re.search(r'AO@([\d.]+)C', line)
                if temp_match: self.stats["temp"] = float(temp_match.group(1))
                
                # 전력 사용량 추출 (POM_5V_IN 기준 mW)
                pwr_match = re.search(r'POM_5V_IN (\d+)/(\d+)', line)
                if pwr_match: self.stats["power"] = int(pwr_match.group(1))

    def stop(self):
        self.running = False

# =============================
# 3. TRT Wrapper 및 벤치마크 루프
# =============================
class TRTWrapper:
    def __init__(self, engine_path):
        self.runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
        with open(engine_path, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            size = abs(trt.volume(self.engine.get_tensor_shape(name))) * 4
            device_mem = cuda.mem_alloc(size)
            self.context.set_tensor_address(name, int(device_mem))
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.input_ptr = device_mem

    def infer(self):
        self.context.execute_async_v3(self.stream.handle)
        self.stream.synchronize()

def run_benchmark(model_path, video_path):
    monitor = SystemMonitor()
    monitor.start()
    
    cap = cv2.VideoCapture(f'filesrc location={video_path} ! qtdemux ! h264parse ! nvv4l2decoder ! nvvidconv ! video/x-raw, format=BGR ! appsink', cv2.CAP_GSTREAMER)
    trt_node = TRTWrapper(model_path)
    
    # CSV 설정
    if not os.path.exists("logs"): os.makedirs("logs")
    log_path = f"logs/jetson_bench_{datetime.now().strftime('%m%d_%H%M')}.csv"
    f = open(log_path, 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(['Frame_ID', 'Timestamp', 'System_FPS', 'E2E_Latency_ms', 'CPU_Usage_%', 'GPU_Usage_%', 'Temp_C', 'Power_mW'])

    in_w, in_h = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out_w, out_h = 640, 384
    d_frame_raw = cuda.mem_alloc(in_h * in_w * 3)
    
    frame_id = 0
    t_total_start = time.time()

    print(f"[INFO] 벤치마크 시작: {log_path}")
    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            t_start = time.perf_counter()

            # 파이프라인 가속
            cuda.memcpy_htod_async(d_frame_raw, frame, trt_node.stream)
            preprocess_kernel(d_frame_raw, trt_node.input_ptr, np.int32(in_w), np.int32(in_h), 
                              np.int32(out_w), np.int32(out_h), block=(16,16,1), 
                              grid=((out_w+15)//16, (out_h+15)//16), stream=trt_node.stream)
            trt_node.infer()

            t_end = time.perf_counter()
            latency = (t_end - t_start) * 1000
            sys_fps = 1000.0 / latency
            
            # 데이터 기록
            writer.writerow([frame_id, datetime.now().strftime("%H:%M:%S.%f"), 
                             f"{sys_fps:.2f}", f"{latency:.2f}", 
                             f"{monitor.stats['cpu']:.1f}", monitor.stats['gpu'], 
                             monitor.stats['temp'], monitor.stats['power']])

            if frame_id % 100 == 0:
                print(f"[{frame_id}] Latency: {latency:.2f}ms | GPU: {monitor.stats['gpu']}% | Power: {monitor.stats['power']}mW")
            frame_id += 1

    finally:
        f.close()
        monitor.stop()
        cap.release()
        avg_fps = frame_id / (time.time() - t_total_start)
        print(f"\n[완료] 평균 시스템 FPS: {avg_fps:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    run_benchmark(parser.parse_args().model, parser.parse_args().video)