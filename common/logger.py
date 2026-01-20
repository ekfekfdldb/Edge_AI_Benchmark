import csv
import time
import psutil
import os
import sys
from datetime import datetime

try:
    from jtop import jtop
    IS_JETSON = True
except ImportError:
    IS_JETSON = False

class ResearchLogger:
    def __init__(self, device_name, model_name, video_name):
        self.device_name = device_name
        
        if not os.path.exists("logs"):
            os.makedirs("logs")
            print("[SYSTEM] 'logs' 폴더가 생성되었습니다.")
        
        safe_model = os.path.basename(model_name)
        date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = f"logs/{date_str}_{device_name}.csv"
        
        try:
            self.file = open(self.log_path, mode='w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
        except PermissionError:
            print(f"[CRITICAL] 파일 쓰기 권한이 없습니다: {self.log_path}")
            sys.exit(1)
        
        header = [
            "Frame_ID", 
            "Timestamp", 
            "E2E_Latency_ms",
            "Inference_Time_ms",
            "CPU_Usage_Percent", 
            "Memory_Usage_Percent",
            "Chip_Temp_C", 
            "Power_W",
            "Model_Name",
            "Video_Source"
        ]
        self.writer.writerow(header)
        
        self.jetson = None
        if IS_JETSON and "Jetson" in device_name:
            print("[SYSTEM] Jetson JTOP 통계 모듈 초기화 중...")
            try:
                self.jetson = jtop()
                self.jetson.start()
            except Exception as e:
                print(f"[WARNING] JTOP 실행 실패 (전력 측정 불가): {e}")

        print(f"[INFO] 로그 기록 시작: {self.log_path}")

    def get_temp(self):
        """칩 코어 온도 측정 (시스템 파일 읽기 방식 - 가장 안전함)"""
        try:
            if self.jetson and self.jetson.ok():
                return self.jetson.stats.get('Temp GPU', 0)
            else:
                with open("/sys/class/thermal/thermal_zone0/temp", "r") as f:
                    temp_millidegree = int(f.read())
                    return temp_millidegree / 1000.0
        except:
            return 0.0

    def get_power(self):
        """소비 전력 측정"""
        try:
            if self.jetson and self.jetson.ok():
                return self.jetson.stats.get('Power TOT', 0) / 1000.0
            return 0.0
        except:
            return 0.0

    def log_frame(self, frame_id, e2e_ms, infer_ms, model_name, video_name):
        try:
            cpu = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory().percent
            temp = self.get_temp()
            power = self.get_power()
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]

            row = [
                frame_id, timestamp, 
                f"{e2e_ms:.4f}", f"{infer_ms:.4f}", 
                cpu, mem, temp, f"{power:.2f}",
                model_name, video_name
            ]
            self.writer.writerow(row)
            
            if frame_id % 50 == 0:
                self.file.flush()
        except Exception as e:
            print(f"[ERROR] 로깅 중 에러 발생: {e}")

    def close(self):
        if self.jetson:
            self.jetson.close()
        if self.file:
            self.file.close()
        print(f"[INFO] 실험 종료. 데이터 저장 완료: {self.log_path}")