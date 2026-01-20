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
except ImportError as e:
    print("\n[CRITICAL] 필수 라이브러리가 없습니다.")
    print(f"에러 내용: {e}")
    print("다음 명령어로 설치하세요: pip3 install pycuda\n")
    sys.exit(1)

from common.logger import ResearchLogger

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

class TRTWrapper:
    def __init__(self, engine_path):
        self.engine_path = engine_path
        self.logger = TRT_LOGGER
        self.runtime = trt.Runtime(self.logger)
        
        try:
            with open(engine_path, "rb") as f:
                self.engine = self.runtime.deserialize_cuda_engine(f.read())
        except Exception as e:
            print(f"[CRITICAL] 엔진 파일을 읽을 수 없습니다. 경로와 파일 상태를 확인하세요.")
            print(f"경로: {engine_path}\n에러: {e}")
            sys.exit(1)
        
        if self.engine is None:
            print("[CRITICAL] 엔진 역직렬화(Deserialize) 실패. TensorRT 버전이 맞는지 확인하세요.")
            sys.exit(1)

        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        self.inputs = []
        self.outputs = []
        self.bindings = []
        self.allocations = []
        
        for binding in self.engine:

            try:
                if hasattr(self.engine, "get_tensor_shape"): 
                    shape = self.engine.get_tensor_shape(binding)
                    dtype = trt.nptype(self.engine.get_tensor_dtype(binding))
                    size = trt.volume(shape) * self.engine.max_batch_size
                else:
                    shape = self.engine.get_binding_shape(binding)
                    dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                    size = trt.volume(shape) * self.engine.max_batch_size
            except:
                 size = 1 * 3 * 640 * 640 
                 dtype = np.float32

            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            
            self.bindings.append(int(device_mem))
            self.allocations.append((host_mem, device_mem))
            
            is_input = False
            if hasattr(self.engine, "get_tensor_mode"):
                if self.engine.get_tensor_mode(binding) == trt.TensorIOMode.INPUT:
                    is_input = True
            elif self.engine.binding_is_input(binding):
                is_input = True

            if is_input:
                self.inputs.append({'host': host_mem, 'device': device_mem})
            else:
                self.outputs.append({'host': host_mem, 'device': device_mem})

    def infer(self, input_image_data):
        try:
            np.copyto(self.inputs[0]['host'], input_image_data.ravel())
            cuda.memcpy_htod_async(self.inputs[0]['device'], self.inputs[0]['host'], self.stream)
            
            self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
            
            for out in self.outputs:
                cuda.memcpy_dtoh_async(out['host'], out['device'], self.stream)
                
            self.stream.synchronize()
        except Exception as e:
            print(f"[ERROR] 추론 중 CUDA 에러 발생: {e}")

def check_file(path):
    if not os.path.exists(path):
        print(f"[CRITICAL] 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

def run_experiment(model_path, video_path):
    check_file(model_path)
    check_file(video_path)

    print(f"[INFO] TensorRT 엔진 로딩 중...: {model_path}")
    trt_wrapper = TRTWrapper(model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[CRITICAL] 영상을 열 수 없습니다: {video_path}")
        sys.exit(1)

    logger = ResearchLogger("Jetson_Orin_Nano", "Universal_TRT", os.path.basename(video_path))
    
    frame_id = 0
    print("[INFO] >>> 실험 시작 (종료하려면 Ctrl+C) <<<")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] 영상 종료. 실험을 마칩니다.")
                break

            target_size = (640, 640) 
            resized = cv2.resize(frame, target_size)
            
            input_data = resized.astype(np.float32) / 255.0
            input_data = input_data.transpose((2, 0, 1))
            input_data = np.expand_dims(input_data, axis=0)

            t_start = time.perf_counter_ns()
            
            trt_wrapper.infer(input_data)
            
            t_end = time.perf_counter_ns()

            e2e_ms = (t_end - t_start) / 1_000_000.0
            
            logger.log_frame(frame_id, e2e_ms, e2e_ms, model_path, video_path)

            if frame_id % 100 == 0:
                print(f"[{frame_id} Frame] Latency: {e2e_ms:.2f}ms | Temp: {logger.get_temp()}C")
            
            frame_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단 (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] 실행 중 치명적 오류 발생: {e}")
    finally:
        cap.release()
        logger.close()
        print("[INFO] 리소스 해제 완료.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help='Path to .engine file')
    parser.add_argument('--video', type=str, required=True, help='Path to video file')
    args = parser.parse_args()

    run_experiment(args.model, args.video)