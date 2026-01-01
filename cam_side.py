import cv2
import numpy as np
import math

confidence = 0.2

def calculate_neck_angle(ear_pt, shoulder_pt):
        """
        輸入: 耳朵座標 (x, y), 肩膀座標 (x, y)
        輸出: 與水平面的夾角 (0~90度)
        """
        ex, ey = ear_pt
        sx, sy = shoulder_pt

        dx = abs(ex - sx)
        dy = abs(sx - sy) # 垂直高度差
    
        if dx == 0: dx = 0.0001 # 避免除以0錯誤

        # 使用 atan2 計算角度 (回傳弧度)
        radian = math.atan2(dy, dx) 
    
        # 轉為角度
        return math.degrees(radian)

class SideSystem:
    def __init__(self):
        self.total_frames = 0
        self.valid_count = 0
        
        # --- 校正參數 (來自您的設定) ---
        self.K = np.array([[640, 0, 480], [0, 640, 240], [0, 0, 1]], dtype=np.float32)
        self.D = np.array([-0.06, -0.2, 0.0, 0.0], dtype=np.float32)

        self.TARGET_ANGLE = 55

    def reset(self):
        """ 重置計數器 """
        self.total_frames = 0
        self.valid_count = 0
        print("[Side] Counters Reset!")

    def undistort(self, frame):
        """ 廣角校正 """
        if frame is None: return None
        return cv2.undistort(frame, self.K, self.D)

    def process(self, image, keypoints_list):
        """
        輸入: 影像, OpenPose 關鍵點
        功能: 鎖定右側 (鏡頭在人體右方)，基準線 55 度
        """
        self.total_frames += 1
        
        is_detected = False
        current_angle = 0 

        if keypoints_list is not None and len(keypoints_list) > 0:
            keypoints = keypoints_list[0]
            
            # --- 強制只讀取右側關鍵點 ---
            # 右耳: 17, 右肩: 2
            target_ear = keypoints[17]
            target_shoulder = keypoints[2]
            
            # 檢查信心分數 (避免雜訊)
            score = target_ear[2] + target_shoulder[2]

            
            # 只有當右側信心足夠時才運算
            if score > confidence:
                ex, ey = int(target_ear[0]), int(target_ear[1])
                sx, sy = int(target_shoulder[0]), int(target_shoulder[1])

                # 防呆 (避免座標 0,0)
                if not ((ex == 0 and ey == 0) or (sx == 0 and sy == 0)):
                    is_detected = True

                    # --- 判斷面向方向 (Direction) ---
                    # 鏡頭在右方，人面向螢幕，通常在畫面中是面向左邊 (Direction = -1)
                    # 邏輯: 如果 耳朵X < 肩膀X，代表面向左 (-1)
                    #       如果 耳朵X > 肩膀X，代表面向右 (1)
                    direction = -1 if ex < sx else 1

                    # --- 1. 計算角度 ---
                    current_angle = calculate_neck_angle((ex, ey), (sx, sy))

                    # --- 2. 判斷標準 ---
                    if current_angle >= self.TARGET_ANGLE:
                        status, color = "Good", (0, 255, 0)
                    elif current_angle >= (self.TARGET_ANGLE - 10): # 45~55
                        status, color = "Warning", (0, 255, 255)
                    else:
                        status, color = "Bad Pos", (0, 0, 255)

                    # --- 3. 繪製 55度 基準線 (灰色) ---
                    ref_len = 150
                    ref_rad = math.radians(self.TARGET_ANGLE)
                    
                    # 計算基準線終點
                    # direction 控制線是往左傾斜還是往右傾斜
                    ref_dx = int(ref_len * math.cos(ref_rad)) * direction
                    ref_dy = int(ref_len * math.sin(ref_rad)) 
                    
                    rx, ry = sx + ref_dx, sy - ref_dy
                    
                    cv2.line(image, (sx, sy), (rx, ry), (150, 150, 150), 2)
                    cv2.putText(image, f"{self.TARGET_ANGLE} deg", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)

                    # --- 4. 繪製實際連線 ---
                    cv2.line(image, (sx, sy), (ex, ey), color, 4)
                    cv2.circle(image, (ex, ey), 8, (255, 0, 0), -1) # 耳朵 (藍點)
                    cv2.circle(image, (sx, sy), 8, (0, 0, 255), -1) # 肩膀 (紅點)

                    # 顯示文字
                    cv2.putText(image, f"Angle: {int(current_angle)}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    cv2.putText(image, status, (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        # --- 統計數據更新 ---
        if is_detected:
            self.valid_count += 1
            
        accuracy = (self.valid_count / self.total_frames * 100) if self.total_frames > 0 else 0

        # --- 顯示 UI ---
        cv2.putText(image, f"Cam 2: Right Side (> {self.TARGET_ANGLE})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, f"Acc: {accuracy:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        return image