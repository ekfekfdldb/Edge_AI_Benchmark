import cv2
import time
import argparse
import sys
import os
import numpy as np

try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
except ImportError:
    sys.exit(1)

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTWrapper:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        
        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except Exception:
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
                self.inputs.append({'host': host_mem, 'device': device_mem, 'shape': shape})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem, 'shape': shape, 'name': name})

    def infer(self, input_data):
        np.copyto(self.inputs[0]['host'], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
        self.context.execute_async_v3(stream_handle=self.stream.handle)
        for out in self.outputs:
            cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
        self.stream.synchronize()
        return [out['host'] for out in self.outputs]

def run_visualization(model_path, video_path):
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        sys.exit(1)

    print(f"[INFO] Load Engine: {model_path}")
    trt_wrapper = TRTWrapper(model_path)
    
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_dir = "output_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(output_dir, f"result_{input_video_name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    print(f"[INFO] Save Path: {save_path}")

    frame_id = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            target_h, target_w = 384, 640
            resized = cv2.resize(frame, (target_w, target_h))
            input_data = (resized.astype(np.float32) / 255.0).transpose(2, 0, 1)
            input_data = np.expand_dims(input_data, axis=0)

            outputs = trt_wrapper.infer(input_data)

            da_seg_mask = None
            
            for output, info in zip(outputs, trt_wrapper.outputs):
                shape = info['shape']
                if len(shape) == 4 and shape[2] == target_h and shape[3] == target_w:
                    seg_map = output.reshape(shape)
                    mask = np.argmax(seg_map, axis=1).squeeze()
                    
                    if da_seg_mask is None:
                        da_seg_mask = mask
                    else:
                        pass
            
            canvas = resized.copy()
            
            if da_seg_mask is not None:
                color_mask = np.zeros_like(canvas)
                color_mask[da_seg_mask == 1] = [0, 255, 0]
                canvas = cv2.addWeighted(canvas, 0.7, color_mask, 0.3, 0)

            final_frame = cv2.resize(canvas, (width, height))
            out_writer.write(final_frame)

            if frame_id % 100 == 0:
                print(f"Processing frame {frame_id}...", flush=True)
            frame_id += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        out_writer.release()
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--video', type=str, required=True)
    args = parser.parse_args()
    
    run_visualization(args.model, args.video)