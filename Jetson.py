import cv2
import time
import argparse
import os
import numpy as np
import csv
import threading
import subprocess
import re
from datetime import datetime

import tensorrt as trt
import pycuda.driver as cuda

cuda.init()

CUDA_CODE = r'''
__global__ void preprocess(unsigned char* input, float* output,
                           int in_w, int in_h, int out_w, int out_h) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= out_w || y >= out_h) return;

    int src_x = x * in_w / out_w;
    int src_y = y * in_h / out_h;
    int in_idx = (src_y * in_w + src_x) * 3;
    int out_plane = out_w * out_h;

    output[y * out_w + x] = input[in_idx + 2] / 255.0f;              // R
    output[out_plane + y * out_w + x] = input[in_idx + 1] / 255.0f; // G
    output[2 * out_plane + y * out_w + x] = input[in_idx + 0] / 255.0f; // B
}
'''

class SystemMonitor(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__()
        self.interval = interval
        self.running = True
        self.daemon = True
        self.stats = {
            "cpu": 0.0,
            "gpu": 0,
            "temp": 0.0,
            "power": 0
        }

    def run(self):
        process = subprocess.Popen(
            ['/usr/bin/tegrastats', '--interval', str(int(self.interval * 1000))],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True
        )

        while self.running:
            line = process.stdout.readline()
            if not line:
                continue

            gpu_match = re.search(r'GR3D_FREQ (\d+)%', line)
            if gpu_match:
                self.stats["gpu"] = int(gpu_match.group(1))

            cpu_match = re.findall(r'(\d+)%@', line)
            if cpu_match:
                self.stats["cpu"] = sum(map(int, cpu_match)) / len(cpu_match)


            for key in ['thermal', 'SOC', 'AO', 'GPU', 'CPU']:
                temp_m = re.search(rf'{key}@([\d.]+)C', line, re.IGNORECASE)
                if temp_m:
                    self.stats["temp"] = float(temp_m.group(1))
                    break


            pwr_match = re.search(r'(VDD_IN|POM_5V_IN)\s+(\d+)mW', line, re.IGNORECASE)
            if pwr_match:
                self.stats["power"] = int(pwr_match.group(2))
            elif 'mW' in line:
                fallback_pwr = re.search(r'(\d+)mW', line)
                if fallback_pwr:
                    self.stats["power"] = int(fallback_pwr.group(1))

        process.terminate()

    def stop(self):
        self.running = False


def run_benchmark(model_path, video_path):
    dev = cuda.Device(0)
    ctx = dev.make_context()

    try:
        from pycuda.compiler import SourceModule
        mod = SourceModule(CUDA_CODE)
        preprocess_kernel = mod.get_function("preprocess")

        monitor = SystemMonitor(interval=0.2)
        monitor.start()
        time.sleep(1.0)

        cap = cv2.VideoCapture(os.path.abspath(video_path))
        ret, first_frame = cap.read()
        if not ret:
            print("[ERROR] 비디오 파일을 읽을 수 없습니다.")
            return

        in_h, in_w, _ = first_frame.shape
        
        logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        with engine.create_execution_context() as context:
            stream = cuda.Stream()
            
            input_name = engine.get_tensor_name(0)
            input_shape = engine.get_tensor_shape(input_name)
            out_h, out_w = input_shape[2], input_shape[3]
            print(f"[INFO] 모델 입력 크기 감지: {out_w}x{out_h}")

            input_ptr = None
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = engine.get_tensor_shape(name)
                size = abs(trt.volume(shape)) * 4 # float32
                device_mem = cuda.mem_alloc(size)
                context.set_tensor_address(name, int(device_mem))
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    input_ptr = device_mem

            os.makedirs("logs", exist_ok=True)
            log_path = f"logs/jetson_final_bench_{datetime.now().strftime('%m%d_%H%M')}.csv"

            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Frame_ID", "Timestamp", "System_FPS",
                    "E2E_Latency_ms", "CPU_Usage_%",
                    "GPU_Usage_%", "Temp_C", "Power_mW"
                ])

                d_frame_raw = cuda.mem_alloc(in_h * in_w * 3)
                curr_frame = first_frame
                frame_id = 0
                t_total_start = time.time()

                print(f"[INFO] 벤치마크 시작: {log_path}")

                while True:
                    t_start = time.perf_counter()

                    cuda.memcpy_htod_async(d_frame_raw, curr_frame, stream)
                    preprocess_kernel(
                        d_frame_raw, input_ptr,
                        np.int32(in_w), np.int32(in_h),
                        np.int32(out_w), np.int32(out_h),
                        block=(16, 16, 1),
                        grid=((out_w + 15) // 16, (out_h + 15) // 16),
                        stream=stream
                    )

                    context.execute_async_v3(stream.handle)
                    stream.synchronize()

                    t_end = time.perf_counter()
                    latency = (t_end - t_start) * 1000

                    writer.writerow([
                        frame_id,
                        datetime.now().strftime("%H:%M:%S.%f"),
                        f"{1000.0 / latency:.2f}",
                        f"{latency:.2f}",
                        f"{monitor.stats['cpu']:.1f}",
                        monitor.stats["gpu"],
                        f"{monitor.stats['temp']:.1f}",
                        monitor.stats["power"]
                    ])

                    if frame_id % 100 == 0:
                        print(
                            f"[{frame_id}] FPS {1000/latency:.1f} | "
                            f"GPU {monitor.stats['gpu']}% | "
                            f"Temp {monitor.stats['temp']:.1f}C | "
                            f"Pwr {monitor.stats['power']}mW"
                        )

                    frame_id += 1
                    ret, curr_frame = cap.read()
                    if not ret:
                        break

        print(f"\n[완료] 평균 시스템 FPS: {frame_id / (time.time() - t_total_start):.2f}")

    except Exception as e:
        print(f"[CRITICAL ERROR] {e}")
    finally:
        monitor.stop()
        cap.release()
        ctx.pop()
        print("[INFO] 자원 해제 완료.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="TensorRT 엔진 파일 경로")
    parser.add_argument("--video", required=True, help="테스트 비디오 파일 경로")
    args = parser.parse_args()
    run_benchmark(args.model, args.video)