import cv2
import time
import argparse
import os
import numpy as np
import csv
import threading
import subprocess
import re
import gc
import queue
from datetime import datetime

import tensorrt as trt
import pycuda.driver as cuda

cuda.init()

ANCHORS = [
    [[3, 9], [5, 11], [4, 20]],
    [[7, 18], [6, 39], [12, 31]],
    [[19, 50], [38, 81], [68, 157]]
]
STRIDES = [8, 16, 32]

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def decode_detections(det_outputs, conf_thres=0.4):
    boxes, scores, class_ids = [], [], []
    for i, pred in enumerate(det_outputs):
        stride = STRIDES[i]
        anchor = np.array(ANCHORS[i])
        bs, h, w, ch = pred.shape
        pred = pred.reshape(bs, h, w, 3, 6)
        pred = sigmoid(pred)
        
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
        grid = np.stack((grid_x, grid_y), axis=2).reshape(1, h, w, 1, 2)

        pred_xy = (pred[..., 0:2] * 2.0 - 0.5 + grid) * stride
        anchor_broadcast = anchor.reshape(1, 1, 1, 3, 2)
        pred_wh = (pred[..., 2:4] * 2.0) ** 2 * anchor_broadcast
        
        pred_conf = pred[..., 4]
        pred_cls = pred[..., 5]
        final_score = pred_conf * pred_cls
        
        mask = final_score > conf_thres
        if not np.any(mask): continue
            
        valid_xy = pred_xy[mask]
        valid_wh = pred_wh[mask]
        valid_scores = final_score[mask]
        
        x1y1 = valid_xy - valid_wh / 2
        valid_boxes = np.concatenate([x1y1, valid_wh], axis=1)
        
        boxes.extend(valid_boxes.tolist())
        scores.extend(valid_scores.tolist())
        class_ids.extend([0] * len(valid_scores))
    return boxes, scores, class_ids

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
    output[y * out_w + x] = input[in_idx + 2] / 255.0f;
    output[out_plane + y * out_w + x] = input[in_idx + 1] / 255.0f;
    output[2 * out_plane + y * out_w + x] = input[in_idx + 0] / 255.0f;
}
'''

def set_performance_mode():
    try:
        subprocess.run(['sh', '-c', 'echo 255 > /sys/devices/pwm-fan/target_pwm'], check=False)
        subprocess.run(['/usr/bin/jetson_clocks'], check=False)
    except: pass

class SystemMonitor(threading.Thread):
    def __init__(self, interval=0.5):
        super().__init__()
        self.interval = interval
        self.running = True
        self.daemon = True
        self.stats = {"gpu": 0, "temp": 0.0, "power": 0}
    def run(self):
        try:
            cmd = ['/usr/bin/tegrastats', '--interval', str(int(self.interval * 1000))]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            while self.running:
                line = process.stdout.readline()
                if not line: break
                pwr_m = re.search(r'(\d+)mW', line)
                if pwr_m: self.stats["power"] = int(pwr_m.group(1))
                temp_m = re.search(r'SOC@([\d.]+)C', line)
                if temp_m: self.stats["temp"] = float(temp_m.group(1))
            process.terminate()
        except: pass
    def stop(self): self.running = False

def run_benchmark(model_path, video_path, mode='sync'):
    set_performance_mode()
    dev = cuda.Device(0)
    ctx = dev.make_context()
    
    logger = trt.Logger(trt.Logger.WARNING)
    with open(model_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())
    
    context = engine.create_execution_context()
    stream = cuda.Stream()
    
    from pycuda.compiler import SourceModule
    mod = SourceModule(CUDA_CODE)
    preprocess_kernel = mod.get_function("preprocess")

    cap = cv2.VideoCapture(os.path.abspath(video_path))
    w0, h0 = int(cap.get(3)), int(cap.get(4))

    input_name = engine.get_tensor_name(0)
    input_shape = tuple(engine.get_tensor_shape(input_name))
    in_h, in_w = input_shape[2], input_shape[3]
    rx, ry = w0 / in_w, h0 / in_h

    d_frame_raw = cuda.mem_alloc(h0 * w0 * 3)
    h_pinned_input = cuda.pagelocked_empty((h0, w0, 3), dtype=np.uint8)
    
    h_outputs = []
    input_ptr = None

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)
        shape = tuple(engine.get_tensor_shape(name))
        size = abs(trt.volume(shape)) * 4
        d_mem = cuda.mem_alloc(size)
        context.set_tensor_address(name, int(d_mem))
        
        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            h_mem = cuda.pagelocked_empty(shape, dtype=np.float32)
            h_outputs.append((name, h_mem, d_mem))
        else:
            input_ptr = d_mem

    monitor = SystemMonitor()
    monitor.start()
    
    MAX_FRAMES = 9000
    log_buffer = [None] * MAX_FRAMES
    frame_id = 0
    gc.disable()
    start_time_global = time.time()

    input_queue = queue.Queue(maxsize=5)
    def producer():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret: break
            input_queue.put(frame)
        input_queue.put(None)

    if mode == 'async':
        threading.Thread(target=producer, daemon=True).start()

    print(f"\n[INFO] Running {mode.upper()} Benchmark with YOLOP Post-processing...")

    try:
        while frame_id < MAX_FRAMES:
            t_e2e_start = time.perf_counter()
            
            if mode == 'sync':
                ret, frame = cap.read()
                if not ret: break
            else:
                frame = input_queue.get()
                if frame is None: break

            t_lat_start = time.perf_counter()
            
            np.copyto(h_pinned_input, frame)
            cuda.memcpy_htod_async(d_frame_raw, h_pinned_input, stream)
            
            preprocess_kernel(d_frame_raw, input_ptr, np.int32(w0), np.int32(h0), 
                              np.int32(in_w), np.int32(in_h), 
                              block=(16, 16, 1), grid=((in_w+15)//16, (in_h+15)//16), stream=stream)
            
            context.execute_async_v3(stream.handle)
            
            for name, h_mem, d_mem in h_outputs:
                cuda.memcpy_dtoh_async(h_mem, d_mem, stream)
            
            stream.synchronize()
            t_lat_end = time.perf_counter()

            output_dict = {name: h_mem for name, h_mem, d_mem in h_outputs}
            
            det_outs = [v for k, v in output_dict.items() if len(v.shape) == 4 and v.shape[3] == 18]
            seg_outs = [v for k, v in output_dict.items() if len(v.shape) == 4 and v.shape[3] in [1, 2]]
            
            det_outs.sort(key=lambda x: x.shape[1], reverse=True) 

            if len(seg_outs) < 2:

                all_outs = [v for k, v in output_dict.items()]
                seg_outs = [v for v in all_outs if v not in det_outs]
            

            da_seg = seg_outs[0] if len(seg_outs) > 0 else np.zeros((1, in_h, in_w, 2))
            ll_seg = seg_outs[1] if len(seg_outs) > 1 else da_seg
            
            da_seg, ll_seg = seg_outs[0], seg_outs[1]
            da_diff = da_seg[0][..., 1] - da_seg[0][..., 0]
            da_mask = (da_diff > 0.0).astype(np.uint8)
            ll_mask = np.argmax(ll_seg[0], axis=-1).astype(np.uint8)

            boxes, scores, _ = decode_detections(det_outs, conf_thres=0.4)
            final_boxes = []
            for b in boxes:
                x, y, w, h = b
                final_boxes.append([int(x*rx), int(y*ry), int(w*rx), int(h*ry)])
            
            _ = cv2.dnn.NMSBoxes(final_boxes, scores, 0.4, 0.45)

            t_e2e_end = time.perf_counter()

            lat_ms = (t_lat_end - t_lat_start) * 1000.0
            e2e_ms = (t_e2e_end - t_e2e_start) * 1000.0
            fps = 1000.0 / e2e_ms
            
            log_buffer[frame_id] = (frame_id, time.time(), fps, lat_ms, e2e_ms, monitor.stats["power"], monitor.stats["temp"])

            if frame_id % 1000 == 0:
                print(f"[{frame_id}] E2E: {e2e_ms:.1f}ms | Pure_Lat: {lat_ms:.1f}ms | FPS: {fps:.1f}")
            
            frame_id += 1

    except KeyboardInterrupt: pass
    finally:
        monitor.stop()
        cap.release()
        gc.enable()
        
        os.makedirs("logs", exist_ok=True)
        filename = f"logs/jetson_yolop_final_{mode}_{datetime.now().strftime('%m%d_%H%M')}.csv"
        with open(filename, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Frame_ID", "Time", "FPS", "Pure_Lat_ms", "E2E_Lat_ms", "Power_mW", "Temp_C"])
            for i in range(frame_id):
                if log_buffer[i]: writer.writerow(log_buffer[i])
        
        print(f"\n[DONE] Results saved to: {filename}")
        ctx.pop()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--video", required=True)
    parser.add_argument("--mode", choices=['sync', 'async'], default='sync')
    args = parser.parse_args()
    run_benchmark(args.model, args.video, args.mode)