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
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                                ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)
except ImportError:
    print("[CRITICAL] 'hailo-platform' 라이브러리가 없습니다.")
    sys.exit(1)



def get_system_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
            return int(f.read().strip()) / 1000.0
    except:
        return -1

def get_gpu_load():
    return 0.0

class StandaloneLogger:
    def __init__(self, model_path, video_path):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = os.path.splitext(os.path.basename(model_path))[0]
        
        if not os.path.exists("logs"):
            os.makedirs("logs")

        self.filename = f"logs/{timestamp}_RPi5_Hailo_{model_name}.csv"
        self.file = open(self.filename, 'w', newline='')
        self.writer = csv.writer(self.file)

        self.writer.writerow([
            'Frame_ID', 'Timestamp', 'FPS', 'E2E_Latency_ms', 'Inference_Time_ms',
            'CPU_Usage_%', 'CPU_Freq_MHz', 'Memory_Usage_%', 'GPU_Usage_%', 'Temperature_C', 'Model'
        ])
        print(f"[INFO] Log file created: {self.filename}")

    def log(self, frame_id, fps, e2e_latency, inf_time, cpu_freq, temp, gpu_usage, model):
        now_time = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        cpu_usage = psutil.cpu_percent(interval=None)
        mem_usage = psutil.virtual_memory().percent
        
        self.writer.writerow([
            frame_id, now_time, f"{fps:.2f}", e2e_latency, inf_time,
            cpu_usage, f"{cpu_freq:.1f}", mem_usage, gpu_usage, temp, model
        ])
        self.file.flush()

    def close(self):
        self.file.close()



class HailoWrapper:
    def __init__(self, hef_path):
        if not os.path.exists(hef_path):
            raise FileNotFoundError(f"HEF file not found: {hef_path}")
            
        self.hef = HEF(hef_path)
        self.target = VDevice()
        
        self.configure_params = ConfigureParams.create_from_hef(
            self.hef, interface=HailoStreamInterface.PCIe)
        self.network_groups = self.target.configure(self.hef, self.configure_params)
        self.network_group = self.network_groups[0]
        
        self.input_vparams = InputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        self.output_vparams = OutputVStreamParams.make_from_network_group(
            self.network_group, quantized=False, format_type=FormatType.FLOAT32)
        
        # Initialize Pipeline
        self.pipeline = InferVStreams(self.network_group, self.input_vparams, self.output_vparams)
        self.pipeline.__enter__()

    def infer(self, input_data):
        return self.pipeline.infer(input_data)

    def close(self):
        self.pipeline.__exit__(None, None, None)


def run_experiment(model_path, video_path):
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("[CRITICAL] Model or Video file not found.")
        sys.exit(1)

    print(f"[INFO] Load Engine: {model_path}")
    
    try:
        hailo_wrapper = HailoWrapper(model_path)
    except Exception as e:
        print(f"[ERROR] Failed to init Hailo: {e}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_path)
    logger = StandaloneLogger(model_path, video_path)

    print("[SYSTEM] Stabilizing...", end="", flush=True)
    for _ in range(5): 
        time.sleep(1)
        print(".", end="", flush=True)
    print(" [OK]")

    print(f"[INFO] Start.")
    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            t_start_e2e = time.perf_counter_ns()

            resized = cv2.resize(frame, (640, 640))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            input_data = np.expand_dims(rgb_frame, axis=0).astype(np.float32)

            t_start_inf = time.perf_counter_ns()
            _ = hailo_wrapper.infer(input_data)
            t_end_inf = time.perf_counter_ns()
            
            t_end_e2e = time.perf_counter_ns()
            
            inf_time = (t_end_inf - t_start_inf) / 1_000_000.0
            e2e_latency = (t_end_e2e - t_start_e2e) / 1_000_000.0
            
            fps = 1000.0 / e2e_latency if e2e_latency > 0 else 0
            
            temp = get_system_temp()
            gpu_load = get_gpu_load()
            
            try:
                cpu_freq = psutil.cpu_freq().current
            except:
                cpu_freq = 0

            logger.log(frame_id, fps, e2e_latency, inf_time, cpu_freq, temp, gpu_load, model_path)

            if frame_id % 100 == 0:
                print(f"[{frame_id}] FPS:{fps:.1f} | Latency:{e2e_latency:.1f}ms | GPU:{gpu_load}% | Temp:{temp}C", flush=True)
            frame_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user.")
    except Exception as e:
        print(f"\n[ERROR] Runtime error: {e}")
    finally:
        cap.release()
        hailo_wrapper.close()
        logger.close()
        print("[INFO] Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .hef file')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    args = parser.parse_args()
    
    run_experiment(args.model, args.video)