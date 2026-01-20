import cv2
import time
import argparse
import sys
import os
from ultralytics import YOLO
from common.logger import ResearchLogger

def check_file(path):
    if not os.path.exists(path):
        print(f"[CRITICAL] 파일을 찾을 수 없습니다: {path}")
        print("경로를 다시 확인해주세요.")
        sys.exit(1)

def run_experiment(model_path, video_path):
    check_file(model_path)
    check_file(video_path)

    print(f"[INFO] 모델 로딩 중 (TensorRT): {model_path}")
    try:
        model = YOLO(model_path, task='detect')
    except Exception as e:
        print(f"[CRITICAL] 모델 로드 실패. TensorRT 엔진 파일이 맞나요? 오류: {e}")
        sys.exit(1)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("[CRITICAL] 영상을 열 수 없습니다.")
        sys.exit(1)

    logger = ResearchLogger(
        device_name="Jetson_Orin_Nano",
        model_name=os.path.basename(model_path),
        video_name=os.path.basename(video_path)
    )

    frame_id = 0
    print("[INFO] >>> 실험 시작 (종료하려면 Ctrl+C) <<<")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] 영상 종료. 실험을 마칩니다.")
                break

            t_start = time.perf_counter_ns()

            results = model(frame, verbose=False)

            t_end = time.perf_counter_ns()

            e2e_latency = (t_end - t_start) / 1_000_000.0
            
            inference_time = results[0].speed['inference']

            logger.log_frame(frame_id, e2e_latency, inference_time, model_path, video_path)

            if frame_id % 100 == 0:
                print(f"[{frame_id} Frame] Latency: {e2e_latency:.2f}ms | Temp: {logger.get_temp()}C")
            
            frame_id += 1

    except KeyboardInterrupt:
        print("\n[INFO] 사용자 중단 (Ctrl+C)")
    except Exception as e:
        print(f"\n[ERROR] 실행 중 치명적 오류 발생: {e}")
    finally:
        cap.release()
        logger.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='models/example.engine', help='Path to .engine file')
    parser.add_argument('--video', type=str, default='videos/example.mp4', help='Path to video file')
    args = parser.parse_args()

    run_experiment(args.model, args.video)