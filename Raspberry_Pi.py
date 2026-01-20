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
    print("pip install hailo-platform 명령어로 설치했는지 확인하세요.")
    sys.exit(1)

from common.logger import ResearchLogger

def check_file(path):
    if not os.path.exists(path):
        print(f"[CRITICAL] 파일을 찾을 수 없습니다: {path}")
        sys.exit(1)

def run_experiment(hef_path, video_path):
    check_file(hef_path)
    check_file(video_path)

    print(f"[INFO] HEF 모델 로딩 중: {hef_path}")
    
    try:
        hef = HEF(hef_path)
    except Exception as e:
        print(f"[CRITICAL] HEF 파일 로드 실패. 파일이 손상되었거나 경로가 틀렸을 수 있습니다. {e}")
        sys.exit(1)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[CRITICAL] 영상을 열 수 없습니다.")
        sys.exit(1)

    logger = ResearchLogger(
        device_name="RPi5_Hailo",
        model_name=os.path.basename(hef_path),
        video_name=os.path.basename(video_path)
    )

    target = VDevice()
    configure_params = ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe)
    network_groups = target.configure(hef, configure_params)
    network_group = network_groups[0]
    
    input_vparams = InputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)
    output_vparams = OutputVStreamParams.make_from_network_group(network_group, quantized=False, format_type=FormatType.FLOAT32)

    frame_id = 0
    print("[INFO] >>> 실험 시작 (Hailo NPU) <<<")

    with InferVStreams(network_group, input_vparams, output_vparams) as infer_pipeline:
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("[INFO] 영상 종료.")
                    break

                resized = cv2.resize(frame, (640, 640))
                rgb_frame = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                input_data = np.expand_dims(rgb_frame, axis=0).astype(np.float32)

                t_start = time.perf_counter_ns()

                infer_results = infer_pipeline.infer(input_data)
                
                t_end = time.perf_counter_ns()

                e2e_latency = (t_end - t_start) / 1_000_000.0
                inference_time = e2e_latency

                logger.log_frame(frame_id, e2e_latency, inference_time, hef_path, video_path)

                if frame_id % 100 == 0:
                    print(f"[{frame_id} Frame] Latency: {e2e_latency:.2f}ms | Temp: {logger.get_temp()}C")
                
                frame_id += 1

        except KeyboardInterrupt:
            print("\n[INFO] 사용자 중단")
        except Exception as e:
            print(f"\n[ERROR] Hailo 추론 중 오류: {e}")
        finally:
            cap.release()
            logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/example.hef', help='Path to .hef file')
    parser.add_argument('--video', type=str, default='videos/example.mp4', help='Path to video file')
    args = parser.parse_args()

    run_experiment(args.model, args.video)