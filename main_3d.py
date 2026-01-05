import cv2
import numpy as np
import helper
from cam_front import FrontSystem
from cam_side import SideSystem

# ==========================================
# 1. 設定區：請填入您的側面鏡頭架設數據
# ==========================================
# 側面鏡頭位置 (單位: 公尺) [前, 左, 上]
SIDE_CAM_POS = np.array([0.028018, 0.575436, 0.220303])

# 側面鏡頭角度 (單位: 度)
# 向下 (Pitch): 20度, 向右 (Yaw): 45度
SIDE_CAM_PITCH = 20 
SIDE_CAM_YAW   = 45 

# ==========================================
# 2. 核心工具：計算投影矩陣 P = K[R|T]
# ==========================================
def get_projection_matrices(K_front, K_side):
    # --- A. 正面鏡頭 (世界原點) ---
    # P1 = K [I | 0]
    P1 = K_front @ np.hstack((np.eye(3), np.zeros((3, 1))))

    # --- B. 側面鏡頭 (計算外參) ---
    # 1. 位置轉換: User(X,Y,Z) -> OpenCV(-Y,-Z,X)
    pos_opencv = np.array([
        -SIDE_CAM_POS[1], # X_cv = -Y_user (向左為Y正 -> OpenCV向右為X正, 故負號)
        -SIDE_CAM_POS[2], # Y_cv = -Z_user (向上為Z正 -> OpenCV向下為Y正, 故負號)
         SIDE_CAM_POS[0]  # Z_cv = X_user  (向前為X正 -> OpenCV向前為Z正)
    ]).reshape(3, 1)

    # 2. 旋轉計算 (Camera Orientation)
    # 向右轉 (Yaw) 通常為負角度 (視座標定義而定)，向下轉 (Pitch) 為正
    ang_yaw = np.radians(-SIDE_CAM_YAW)
    ang_pitch = np.radians(SIDE_CAM_PITCH)

    # 旋轉矩陣 (Yaw -> Pitch)
    Ry = np.array([
        [np.cos(ang_yaw), 0, np.sin(ang_yaw)],
        [0, 1, 0],
        [-np.sin(ang_yaw), 0, np.cos(ang_yaw)]
    ])
    Rx = np.array([
        [1, 0, 0],
        [0, np.cos(ang_pitch), -np.sin(ang_pitch)],
        [0, np.sin(ang_pitch), np.cos(ang_pitch)]
    ])
    
    R_cam = Rx @ Ry
    
    # 計算外參矩陣 R (World -> Camera)
    R = R_cam.T
    # 計算外參向量 T (World -> Camera) => T = -R * C
    T = -R @ pos_opencv

    # P2 = K [R | T]
    P2 = K_side @ np.hstack((R, T))
    
    return P1, P2

# ==========================================
# 3. 核心函式：3D 三角測量 (Triangulation)
# ==========================================
def estimate_3d_pose_triangulation(kp_front, kp_side, P1, P2):
    """
    輸入: 
      - kp_front, kp_side: OpenPose 的關鍵點
      - P1, P2: 兩個鏡頭的投影矩陣
    輸出: 
      - points_3d: 字典 {id: (x, y, z)} (單位: 公分)
    """
    points_3d = {}
    
    if (kp_front is None or len(kp_front) == 0) or \
       (kp_side is None or len(kp_side) == 0):
        return None

    # 取得第一個人
    p1 = kp_front[0]
    p2 = kp_side[0]
    
    # 收集有效點的座標
    pts1 = []
    pts2 = []
    indices = []

    for i in range(25):
        fx, fy, fc = p1[i]
        sx, sy, sc = p2[i]

        # 信心度門檻
        if fc > 0.1 and sc > 0.1:
            pts1.append([fx, fy])
            pts2.append([sx, sy])
            indices.append(i)

    if len(pts1) == 0:
        return None

    # 轉為 numpy 格式 (2, N)
    pts1 = np.array(pts1).T.astype(float)
    pts2 = np.array(pts2).T.astype(float)

    # --- 關鍵一步：OpenCV 三角測量 ---
    points_4d = cv2.triangulatePoints(P1, P2, pts1, pts2)

    # 齊次座標歸一化 (X, Y, Z, W) -> (X/W, Y/W, Z/W)
    points_3d_raw = points_4d[:3] / points_4d[3]

    # 轉換回您的座標系定義 & 存入字典
    for k, idx in enumerate(indices):
        x_raw = points_3d_raw[0, k] # OpenCV X (右)
        y_raw = points_3d_raw[1, k] # OpenCV Y (下)
        z_raw = points_3d_raw[2, k] # OpenCV Z (前)

        # 座標轉換: OpenCV -> User (前, 左, 上)
        # User X (前) = OpenCV Z
        # User Y (左) = -OpenCV X
        # User Z (上) = -OpenCV Y
        
        user_x = z_raw
        user_y = -x_raw
        user_z = -y_raw

        # 單位換算: 公尺 -> 公分 (方便閱讀)
        points_3d[idx] = (user_x * 100, user_y * 100, user_z * 100)
            
    return points_3d

# ==========================================
# 4. 主程式
# ==========================================
def main():
    print("正在載入 OpenPose 模型...")
    op_wrapper, op_lib = helper.init_openpose()
    
    print("正在開啟鏡頭...")
    front_sys = FrontSystem(0)
    side_sys = SideSystem(2)  # 注意：請確認您的側面鏡頭 ID 是不是 2

    # --- [NEW] 計算投影矩陣 ---
    # 提醒：請記得更新 cam_front.py 和 cam_side.py 裡面的 self.K
    # 填入您截圖中新算出來的 749.70 等數值，準確度才會最高！
    print("正在計算投影矩陣...")
    P1, P2 = get_projection_matrices(front_sys.K, side_sys.K)

    print("--- 3D 姿態估測系統 (Triangulation版) 啟動 ---")
    print(f"側面鏡頭設定: Pos={SIDE_CAM_POS}, Angle={SIDE_CAM_PITCH}/{SIDE_CAM_YAW}")

    while True:
        # A. 讀取與校正
        ret1, raw1 = front_sys.cap.read()
        ret2, raw2 = side_sys.cap.read()
        
        if not ret1 or not ret2:
            print("遺失鏡頭訊號")
            continue

        frame1 = cv2.undistort(raw1, front_sys.K, front_sys.D)
        frame2 = cv2.undistort(raw2, side_sys.K, side_sys.D)

        # B. 取得關鍵點
        kp1, rendered1 = helper.get_keypoints(op_wrapper, op_lib, frame1)
        kp2, rendered2 = helper.get_keypoints(op_wrapper, op_lib, frame2)

        img1 = rendered1 if rendered1 is not None else frame1
        img2 = rendered2 if rendered2 is not None else frame2

        # --- C. [NEW] 執行 3D 運算 ---
        # 傳入 P1, P2 進行正規計算
        pose_3d = estimate_3d_pose_triangulation(kp1, kp2, P1, P2)

        if pose_3d:
            # 取得 頸部 (Index 1)
            if 1 in pose_3d:
                neck_x, neck_y, neck_z = pose_3d[1]
                
                # 顯示數據 (單位: cm)
                # X: 前後距離 (越大代表越遠)
                # Z: 高度 (越大代表越高，注意這裡是"上"為正)
                info = f"Neck: Dist={neck_x:.1f}cm H={neck_z:.1f}cm"
                cv2.putText(img1, info, (10, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)

                # --- D. 姿態判斷邏輯 ---
                # 判斷標準：看 X (前後距離) 是否變化
                # 假設標準坐姿距離鏡頭 80cm
                standard_dist = 80.0
                threshold = 15.0 # 允許誤差 15cm
                
                diff = neck_x - standard_dist
                
                status = "Normal"
                color = (0, 255, 0)
                
                if diff > threshold: # 距離變大 (往後靠)
                    status = "Backward"
                    color = (0, 255, 255)
                elif diff < -threshold: # 距離變小 (往前傾)
                    status = "Turtle Neck (Forward)"
                    color = (0, 0, 255)

                cv2.putText(img1, f"Status: {status}", (10, 440), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        # E. 畫面拼接
        if img1.shape != img2.shape:
            img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
        combined = np.hstack((img1, img2))
        cv2.line(combined, (640, 0), (640, 480), (255, 255, 255), 2)
        
        cv2.imshow("3D Posture Estimation (Stereo)", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    front_sys.release()
    side_sys.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()