import cv2
import numpy as np
import helper
import time

def main():
    # 1. 初始化 OpenPose (開啟 3D 模式)
    # 請確保 'camera_parameters' 資料夾內有 0.xml 和 1.xml
    print("正在啟動 OpenPose 3D 模組...")
    op_wrapper, op_lib = helper.init_openpose()
    
    # 2. 開啟鏡頭 (直接使用 VideoCapture，不需要額外的校正 Class)
    # 請確認 ID: 0 是正面 (對應 0.xml), 2 是側面 (對應 1.xml)
    cap1 = cv2.VideoCapture(0) 
    cap2 = cv2.VideoCapture(2) 

    # 設定解析度 (必須跟校正時一樣，通常是 640x480)
    for cap in [cap1, cap2]:
        cap.set(3, 640)
        cap.set(4, 480)

    if not cap1.isOpened() or not cap2.isOpened():
        print("無法開啟鏡頭，請檢查 ID")
        return

    print("--- OpenPose Native 3D 系統啟動 ---")
    print("按 'q' 離開")

    while True:
        # 3. 讀取原始影像 (Raw Image)
        # 注意：不要自己做 undistort！OpenPose 會根據 XML 自己做！
        ret1, frame1 = cap1.read()
        ret2, frame2 = cap2.read()
        
        if not ret1 or not ret2:
            print("遺失鏡頭訊號")
            continue

        # 4. 準備 OpenPose 輸入數據 (VectorDatum)
        # 3D 模式必須「同時」塞入多個視角的影像
        datum1 = op_lib.Datum()
        datum1.cvInputData = frame1
        
        datum2 = op_lib.Datum()
        datum2.cvInputData = frame2
        
        # 建立 Vector (列表)
        vector_datum = op_lib.VectorDatum([datum1, datum2])
        
        # 5. 推論與 3D 重建
        op_wrapper.emplaceAndPop(vector_datum)

        # 6. 取得結果
        # keypoints3d 格式: [人數, 25個關節, 4] (x, y, z, score)
        # 這裡的座標單位通常是 mm 或 m (取決於校正時棋盤格的大小設定)
        kp_3d = datum1.poseKeypoints3D 
        
        # 取得 OpenPose 幫我們畫好骨架的圖 (已去畸變)
        img1_out = datum1.cvOutputData
        img2_out = datum2.cvOutputData
        
        # 確保畫面不為 None
        if img1_out is None: img1_out = frame1
        if img2_out is None: img2_out = frame2

        # 7. 數據分析
        if kp_3d is not None and len(kp_3d) > 0:
            # 取得第一個人的 3D 骨架
            person_3d = kp_3d[0] 
            
            # 取得頸部 (Neck, ID=1) 的座標
            # OpenPose 3D 座標系通常是：
            # X: 左右, Y: 上下, Z: 深度 (以 0.xml 鏡頭為基準)
            neck = person_3d[1]
            x, y, z, score = neck[0], neck[1], neck[2], neck[3]

            if score > 0.0: # 有抓到點
                # 顯示數據 (假設單位是 mm，轉成 cm 顯示)
                # 您校正時如果棋盤格是給 mm，這裡出來就是 mm
                info_text = f"Neck 3D: X={x/10:.1f} Y={y/10:.1f} Z={z/10:.1f} cm"
                
                cv2.putText(img1_out, info_text, (10, 400), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

                # --- 簡單的駝背判斷 ---
                # 這裡的 Z 是相對於「正面鏡頭」的距離
                # 如果人往後躺，Z 會變大；往前傾，Z 會變小
                # 請根據實際情況調整 threshold
                standard_z_mm = 800.0 # 假設標準坐姿離鏡頭 80cm
                threshold = 100.0     # 容許範圍 10cm
                
                if z < (standard_z_mm - threshold):
                    status = "Forward (Turtle)"
                    color = (0, 0, 255)
                elif z > (standard_z_mm + threshold):
                    status = "Backward"
                    color = (0, 255, 255)
                else:
                    status = "Normal"
                    color = (0, 255, 0)
                    
                cv2.putText(img1_out, status, (10, 440), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

        # 8. 顯示畫面
        # 拼接兩個畫面方便觀察
        if img1_out.shape == img2_out.shape:
            combined = np.hstack((img1_out, img2_out))
        else:
            combined = img1_out # 尺寸不合就只秀第一張
            
        cv2.imshow("OpenPose Native 3D", combined)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap1.release()
    cap2.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()