import os
import sys

# 取得 visualize.py 所在的目錄路徑
module_path = os.path.dirname(os.path.abspath(__file__))
# 將模組所在的目錄路徑加入 Python 的搜尋路徑
sys.path.append(module_path)

import os
import re
import threading
import warnings
from datetime import datetime

import cv2
import genderAge as modelG
import matplotlib
import visualize as model
from flask import Flask, render_template, request, session

warnings.filterwarnings("ignore")

matplotlib.use("Agg")

UPLOAD_FOLDER = "app/static/uploaded"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static",
    static_url_path="/static",
)

app.secret_key = "your_secret_key_here"


def convert_video_to_h264(input_file, output_file, output_frame_rate):
    # 讀取影片檔案
    cap = cv2.VideoCapture(input_file)

    # 獲取影片的基本資訊
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
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


def process_video(video_path, save_path):
    cap = cv2.VideoCapture(video_path)

    # 抓出影片檔名
    video_filename = os.path.basename(video_path)
    video_filename = os.path.splitext(video_filename)[0]  # 移除副檔名部分

    # 確認影片成功開啟
    if not cap.isOpened():
        print("無法開啟影片檔案")
        return
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
    num_segments = int(duration / segment_length)  # 總段數

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
                if j < 5:
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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/inner-page", methods=["GET", "POST"])
def upload():
    # 在進入 upload() 時清空 session
    session.clear()

    if request.method == "GET":
        return render_template("inner-page.html")
    elif request.method == "POST":
        file = request.files["video"]
        if "report" not in session:
            session["report"] = []
        if file:
            # 使用當前時間來生成唯一的檔案名稱
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
            session["timestamp"] = timestamp

            filename = f"{timestamp}.mp4"
            file.save(os.path.join(UPLOAD_FOLDER, filename))

            # 進行編碼格式轉換
            input_file = os.path.join(UPLOAD_FOLDER, filename)
            output_file = os.path.join("app/static/output_video/", f"{timestamp}.mp4")
            convert_video_to_h264(input_file, output_file, 30)

            # 呼叫 process_video 函數處理影片
            video_path = os.path.join("app/static/output_video/", f"{timestamp}.mp4")
            save_path = "app/static/cutting_img/"  # 設定儲存擷取影片禎數的目錄路徑
            process_video(video_path, save_path)

            # model
            for n in range(5):
                for m in range(5):
                    # 取得 cutting_img 目錄下的圖片檔案列表
                    rec_filename = f"{timestamp}_segment_{n+1}_frame_{m+1}.jpg"
                    try:
                        model.pred_faceExp(rec_filename)
                        break  # 找到第一張符合條件的照片後，跳出 for m in range(5) 迴圈
                    except FileNotFoundError:
                        continue

        return render_template("recog-result.html", session=session)

        # return redirect(url_for("recog_result"))

    else:
        return "檔案上傳失敗"


@app.route("/recog-result", methods=["GET"])
def recog_result():
    if request.method == "GET":
        timestamp = session.get("timestamp")  # 從session中取得 timestamp 變數
        print(timestamp)
        pattern = re.compile(rf"{timestamp}_segment_\d+_frame_\d+\.jpg")
        matching_files = [
            filename
            for filename in os.listdir("app/static/results_img")
            if pattern.match(filename)
        ]
        matching_files = sorted(matching_files)

        # 創建一個新的線程來執行模型的預測
        prediction_thread = threading.Thread(
            target=process_predictions, args=(timestamp,)
        )
        prediction_thread.start()

        return render_template("recog-result.html", report=matching_files)


def process_predictions(timestamp):
    for n in range(5):
        for m in range(5):
            # 取得 cutting_img 目錄下的圖片檔案列表
            pred_filename = f"{timestamp}_segment_{n+1}_frame_{m+1}.jpg"
            try:
                modelG.genderAge(pred_filename)
                break  # 找到第一張符合條件的照片後，跳出 for m in range(5) 迴圈
            except FileNotFoundError:
                continue
