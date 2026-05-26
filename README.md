# 0526

這是一個 MediaPipe Pose 影像與 webcam 偵測專案。

## 專案內容

- `pose_example.py` - MediaPipe Pose Python 範例與遊戲狀態整合。
- `webapp/index.html` - MediaPipe FaceMesh 表情偵測網頁。
- `webapp/server.py` - 本機靜態伺服器，用於在 `http://127.0.0.1:8000` 開啟網頁。
- `requirements.txt` - Python 相依套件。
- `url.txt` - 專案儲存庫網址。

## 使用方式

1. 安裝相依套件：

```bash
python3 -m pip install -r requirements.txt
```

2. 啟動本機網頁伺服器：

```bash
cd webapp
python3 server.py
```

3. 開啟瀏覽器並前往：

```text
http://127.0.0.1:8000/
```

4. 允許攝影機存取，即可啟動 webcam 姿態偵測。
