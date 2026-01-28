import cv2
import time
import argparse
import os
import numpy as np
import csv
import threading
import subprocess
import re
import queue
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

    // BGR -> RGB 및 0~1 정규화 (모델에 따라 조정 가능)
    output[y * out_w + x] = input[in_idx + 2] / 255.0f;             // R
    output[out_plane + y * out_w + x] = input[in_idx + 1] / 255.0f; // G
    output[2 * out_plane + y * out_w + x] = input[in_idx + 0] / 255.0f; // B
}
'''

class VideoStreamReader(threading.Thread):
    def __init__(self, video_path):
        super().__init__()
        self.cap = cv2.VideoCapture(os.path.abspath(video_path))
        
        self.q = queue.Queue(maxsize=1) 
        
        self.running = True
        self.daemon = True
        
        ret, frame = self.cap.read()
        if ret:
            self.first_frame = frame
            self.in_h, self.in_w, _ = frame.shape
        else:
            raise RuntimeError(f"[ERROR] 영상을 열 수 없습니다: {video_path}")

    def run(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            
            capture_ts = datetime.now()

            if self.q.full():
                try:
                    self.q.get_nowait()
                except queue.Empty:
                    pass

            self.q.put((frame, capture_ts))
            
        self.cap.release()

    def get_latest_frame(self):
        try:
            return self.q.get(timeout=1.0)
        except queue.Empty:
            return None

    def stop(self):
        self.running = False

class SystemMonitor(threading.Thread):
    def __init__(self, interval=0.2):
        super().__init__()
        self.interval = interval
        self.running = True
        self.daemon = True
        self.stats = {"cpu": 0.0, "gpu": 0, "temp": 0.0, "power": 0}

    def run(self):
        cmd = ['/usr/bin/tegrastats', '--interval', str(int(self.interval * 1000))]
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True
        )
        
        while self.running:
            line = process.stdout.readline()
            if not line: continue
            
            gpu_m = re.search(r'GR3D_FREQ (\d+)%', line)
            if gpu_m: self.stats["gpu"] = int(gpu_m.group(1))
            

            cpu_m = re.findall(r'(\d+)%@', line)
            if cpu_m: self.stats["cpu"] = sum(map(int, cpu_m)) / len(cpu_m)

            for key in ['thermal', 'SOC', 'AO', 'GPU', 'CPU']:
                temp_m = re.search(rf'{key}@([\d.]+)C', line, re.IGNORECASE)
                if temp_m: 
                    self.stats["temp"] = float(temp_m.group(1))
                    break
            
            pwr_m = re.search(r'(VDD_IN|POM_5V_IN)\s+(\d+)mW', line, re.IGNORECASE)
            if pwr_m: self.stats["power"] = int(pwr_m.group(2))
            elif 'mW' in line:
                fb_pwr = re.search(r'(\d+)mW', line)
                if fb_pwr: self.stats["power"] = int(fb_pwr.group(1))

        process.terminate()

    def stop(self):
        self.running = False

def run_benchmark(model_path, video_path):
    dev = cuda.Device(0)
    ctx = dev.make_context()

    reader = None
    monitor = None

    try:
        from pycuda.compiler import SourceModule
        mod = SourceModule(CUDA_CODE)
        preprocess_kernel = mod.get_function("preprocess")

        print(f"[INFO] 비디오 로드 중: {video_path}")
        reader = VideoStreamReader(video_path)
        in_h, in_w = reader.in_h, reader.in_w
        reader.start()

        monitor = SystemMonitor(interval=0.1)
        monitor.start()
        time.sleep(1.0)

        logger = trt.Logger(trt.Logger.WARNING)
        with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
            engine = runtime.deserialize_cuda_engine(f.read())

        print("[INFO] TensorRT 엔진 로드 완료")

        with engine.create_execution_context() as context:
            stream = cuda.Stream()
            
            input_name = engine.get_tensor_name(0)
            input_shape = engine.get_tensor_shape(input_name)
            out_h, out_w = input_shape[2], input_shape[3] # NCHW 가정
            
            print(f"[INFO] 모델 입력 해상도: {out_w}x{out_h}")

            input_ptr = None
            for i in range(engine.num_io_tensors):
                name = engine.get_tensor_name(i)
                shape = engine.get_tensor_shape(name)
                size = abs(trt.volume(shape)) * 4 # float32 bytes
                device_mem = cuda.mem_alloc(size)
                context.set_tensor_address(name, int(device_mem))
                
                if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                    input_ptr = device_mem

            d_frame_raw = cuda.mem_alloc(in_h * in_w * 3)


            os.makedirs("logs", exist_ok=True)
            log_path = f"logs/bench_final_{datetime.now().strftime('%m%d_%H%M')}.csv"
            
            with open(log_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Frame_ID", "Timestamp", "System_FPS", 
                    "Inference_Latency_ms", "Total_Delay_ms", 
                    "CPU_Usage_%", "GPU_Usage_%", "Temp_C", "Power_mW"
                ])

                frame_id = 0
                t_bench_start = time.time()
                
                print(f"[INFO] 🚀 벤치마크 시작! (File: {log_path})")

                while reader.running:
                    data = reader.get_latest_frame()
                    if data is None:
                        if not reader.running: break
                        continue

                    curr_frame, capture_ts = data
                    
                    t_infer_start = time.perf_counter()

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
                    t_infer_end = time.perf_counter()
                    
                    process_end_ts = datetime.now()

                    infer_latency = (t_infer_end - t_infer_start) * 1000
                    
                    total_delay = (process_end_ts - capture_ts).total_seconds() * 1000
                    
                    fps = 1000.0 / infer_latency if infer_latency > 0 else 0

                    writer.writerow([
                        frame_id,
                        process_end_ts.strftime("%H:%M:%S.%f"),
                        f"{fps:.2f}",
                        f"{infer_latency:.2f}",
                        f"{total_delay:.2f}",
                        f"{monitor.stats['cpu']:.1f}",
                        monitor.stats["gpu"],
                        f"{monitor.stats['temp']:.1f}",
                        monitor.stats["power"]
                    ])

                    if frame_id % 100 == 0:
                        print(
                            f"[{frame_id}] FPS: {fps:.1f} | "
                            f"Delay: {total_delay:.1f}ms | "
                            f"GPU: {monitor.stats['gpu']}% | "
                            f"Pwr: {monitor.stats['power']}mW"
                        )
                    
                    frame_id += 1

                total_time = time.time() - t_bench_start
                avg_fps = frame_id / total_time
                print(f"\n[완료] 평균 시스템 FPS: {avg_fps:.2f}")

    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        if reader: reader.stop()
        if monitor: monitor.stop()
        try:
            ctx.pop()
        except:
            pass
        print("[INFO] 자원 해제 및 종료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="TensorRT 엔진 파일 경로 (.engine)")
    parser.add_argument("--video", required=True, help="테스트 비디오 파일 경로 (.mp4)")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"[오류] 모델 파일을 찾을 수 없습니다: {args.model}")
    elif not os.path.exists(args.video):
        print(f"[오류] 비디오 파일을 찾을 수 없습니다: {args.video}")
    else:
        run_benchmark(args.model, args.video)