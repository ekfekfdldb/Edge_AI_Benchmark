import cv2
import time
import argparse
import sys
import os
import numpy as np

try:
    from hailo_platform import (HEF, VDevice, HailoStreamInterface, InferVStreams, 
                                ConfigureParams, InputVStreamParams, OutputVStreamParams, FormatType)
except ImportError:
    print("[CRITICAL] 'hailo-platform' 라이브러리가 없습니다.")
    sys.exit(1)

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
        
        self.pipeline = InferVStreams(self.network_group, self.input_vparams, self.output_vparams)
        self.pipeline.__enter__()

    def infer(self, input_data):
        return self.pipeline.infer(input_data)

    def close(self):
        self.pipeline.__exit__(None, None, None)

def run_visualization(model_path, video_path):
    if not os.path.exists(model_path) or not os.path.exists(video_path):
        print("[CRITICAL] 모델 또는 영상 파일이 없습니다.")
        sys.exit(1)

    print(f"[INFO] Load Model: {model_path}")
    
    try:
        hailo_wrapper = HailoWrapper(model_path)
    except Exception as e:
        print(f"[ERROR] Hailo 초기화 실패: {e}")
        sys.exit(1)
        
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    output_dir = "output_videos"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    input_video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_path = os.path.join(output_dir, f"result_rpi_{input_video_name}.mp4")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_writer = cv2.VideoWriter(save_path, fourcc, fps, (width, height))
    
    print(f"[INFO] Save Path: {save_path}")

    frame_id = 0

    target_w, target_h = 640, 640 

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break

            resized = cv2.resize(frame, (target_w, target_h))
            rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            
            input_data = np.expand_dims(rgb_frame, axis=0).astype(np.float32) / 255.0

            results = hailo_wrapper.infer(input_data)

            da_seg_mask = None

            for name, output in results.items():
                shape = output.shape

                if len(shape) == 4:
                    h, w = shape[1], shape[2] 
                    
                    if h == target_h and w == target_w:

                        mask = np.argmax(output, axis=3).squeeze() 
                        da_seg_mask = mask
                        break

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
        print("\n[INFO] 사용자 중단")
    except Exception as e:
        print(f"\n[ERROR] 실행 중 오류: {e}")
    finally:
        cap.release()
        out_writer.release()
        hailo_wrapper.close()
        print("Done.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, required=True, help="Path to .hef file")
    parser.add_argument('--video', type=str, required=True, help="Path to video file")
    args = parser.parse_args()
    
    run_visualization(args.model, args.video)