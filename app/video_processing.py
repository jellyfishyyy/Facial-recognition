import os

import cv2


# 錄影檔編碼格式轉換
def convert_video_to_h264(input_file, output_file, output_frame_rate):
    # 讀取影片檔案
    cap = cv2.VideoCapture(input_file)

    # 獲取影片的基本資訊
    # frame_rate = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # 使用 MPEG-4 編碼格式進行轉換
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_file, fourcc, output_frame_rate, (width, height))

    # 讀取並寫入每一禎影像
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    # 釋放資源並關閉影片檔案
    cap.release()
    out.release()

    return cap.get(cv2.CAP_PROP_POS_MSEC) / 1000  # 影片時長（秒）


# Cutting
def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)

    # 抓出影片檔名
    video_filename = os.path.basename(video_path)
    video_filename = os.path.splitext(video_filename)[0]  # 移除副檔名部分

    # 確認影片成功開啟
    if not cap.isOpened():
        print("無法開啟影片檔案")
        return
    # cascade classifier
    face_cascade = cv2.CascadeClassifier(
        "app/model/haarcascade_frontalface_default.xml"
    )
    # 取得影片資訊
    frame_rate = cap.get(cv2.CAP_PROP_FPS)  # 影片的幀率
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))  # 影片的總幀數
    duration = total_frames / frame_rate  # 影片的長度（秒）
    print("影片幀率= ", frame_rate)
    print("影片總幀數= ", total_frames)
    print("影片長度= ", duration)

    # 計算每段影片的開始和結束幀數
    segment_length = 1.7  # 每段影片的長度（秒）
    segment_frames = int(frame_rate * segment_length)  # 每段影片的幀數
    # num_segments = int(duration / segment_length)  # 總段數

    # 處理每一段影片
    for i in range(5):
        # 設定影片指標位置
        start_frame = i * segment_frames + 5.5 * frame_rate
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        # 處理每一禎影片
        for j in range(segment_frames):
            ret, frame = cap.read()
            if ret:
                # 只保存每段影片的前 5 禎
                if j <= 35 & j >= 31:
                    # 人臉偵測
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    faces = face_cascade.detectMultiScale(
                        gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
                    )

                    # 擷取人臉並保存圖片
                    for x, y, w, h in faces:
                        face_image = frame[y : y + h, x : x + w]
                        # 生成儲存的檔案路徑
                        output_path = os.path.join(
                            save_path, f"{video_filename}_segment_{i+1}_frame_{j+1}.jpg"
                        )
                        # 將人臉圖片儲存為 jpg 檔案
                        cv2.imwrite(output_path, face_image)
            else:
                print(f"無法讀取第{i+1}段第{j+1}禎影片")

    # 釋放資源並關閉影片檔案
    cap.release()
