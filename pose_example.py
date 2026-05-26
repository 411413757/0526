import cv2
import mediapipe as mp
import numpy as np
from collections import deque
from pathlib import Path
from typing import Optional

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils


class PoseEstimator:
    """
    封裝 MediaPipe Pose 姿態估測器，管理攝影機串流生命週期。
    支援 Python Context Manager（with 語法）以確保資源正確釋放。
    """

    def __init__(
        self,
        camera_index: int = 0,
        model_complexity: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5
    ) -> None:
        self.camera_index = camera_index
        self._pose = mp_pose.Pose(
            static_image_mode=False,
            model_complexity=model_complexity,
            smooth_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self._cap: Optional[cv2.VideoCapture] = None

    def open_camera(self) -> None:
        """開啟攝影機裝置。"""
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            raise RuntimeError(f"無法開啟攝影機（索引：{self.camera_index}）")
        print(f"✓ 攝影機已開啟（索引：{self.camera_index}）")

    def release(self) -> None:
        """釋放攝影機與 MediaPipe 資源。"""
        if self._cap:
            self._cap.release()
        self._pose.close()
        cv2.destroyAllWindows()
        print("✓ 攝影機資源已釋放。")

    def read_frame(self) -> Optional[np.ndarray]:
        """
        從攝影機讀取一幀並進行水平翻轉。

        Returns:
            BGR 影像幀，若讀取失敗則返回 None
        """
        if not self._cap:
            return None
        success, frame = self._cap.read()
        if not success:
            return None
        return cv2.flip(frame, 1)  # 水平鏡像

    def process_frame(self, frame: np.ndarray):
        """
        對輸入畫面進行姿態估測並繪製骨架。

        Args:
            frame: BGR 影像幀

        Returns:
            (annotated_frame, pose_landmarks): 標注後的畫面與關鍵點物件
        """
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_rgb.flags.writeable = False
        results = self._pose.process(frame_rgb)
        frame_rgb.flags.writeable = True

        annotated = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                annotated,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 255, 0), thickness=2, circle_radius=3
                ),
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    color=(0, 128, 255), thickness=2
                )
            )
        return annotated, results.pose_landmarks

    def __enter__(self):
        self.open_camera()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.release()
        return False


class MathUtils:
    """
    姿態數學運算工具類（全靜態方法）。
    封裝正規化、相似度計算與訊號濾波等數學操作。
    """

    @staticmethod
    def normalize_pose(landmarks) -> np.ndarray | None:
        """從 MediaPipe 關鍵點列表提取並正規化姿態特徵向量。"""
        coords = np.array(
            [[lm.x, lm.y] for lm in landmarks],
            dtype=np.float32
        )
        left_hip = coords[mp_pose.PoseLandmark.LEFT_HIP.value]
        right_hip = coords[mp_pose.PoseLandmark.RIGHT_HIP.value]
        origin = (left_hip + right_hip) / 2.0
        coords -= origin

        scale = np.max(np.linalg.norm(coords, axis=1))
        if scale < 1e-6:
            return None

        coords /= scale
        return coords.flatten()

    @staticmethod
    def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """計算兩姿態向量的餘弦相似度，返回 [-1.0, 1.0]。"""
        denom = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
        if denom < 1e-8:
            return 0.0
        return float(np.clip(np.dot(vec_a, vec_b) / denom, -1.0, 1.0))

    @staticmethod
    def similarity_to_score(similarity: float) -> int:
        """將餘弦相似度線性映射為 0~100 分數。"""
        return int(np.clip((similarity + 1.0) / 2.0 * 100.0, 0, 100))

    @staticmethod
    def moving_average_filter(buffer: "deque[float]",
                              new_value: float,
                              window_size: int = 10) -> float:
        """
        簡單移動平均濾波器（SMA），用於平滑化分數抖動。

        Args:
            buffer: 歷史分數緩衝佇列（deque），直接修改
            new_value: 最新分數
            window_size: 移動平均視窗大小（幀數）

        Returns:
            平滑化後的分數
        """
        buffer.append(new_value)
        if len(buffer) > window_size:
            buffer.popleft()
        return float(np.mean(list(buffer)))


def normalize_pose(landmarks) -> np.ndarray | None:
    """
    封裝呼叫 MathUtils.normalize_pose 的便利函式。
    """
    return MathUtils.normalize_pose(landmarks)


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    封裝呼叫 MathUtils.cosine_similarity 的便利函式。
    """
    return MathUtils.cosine_similarity(vec_a, vec_b)


def similarity_to_score(similarity: float) -> int:
    """
    計算兩個正規化姿態特徵向量的餘弦相似度。

    Args:
        vec_a: 目標姿勢特徵向量 (shape: (66,))
        vec_b: 即時姿勢特徵向量 (shape: (66,))

    Returns:
        相似度分數，範圍 [-1.0, 1.0]
    """
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    denominator = norm_a * norm_b
    if denominator < 1e-8:
        return 0.0
    similarity = dot_product / denominator
    return float(np.clip(similarity, -1.0, 1.0))


PASS_THRESHOLD = 80  # 通關分數閾值（百分制）
PASS_HOLD_FRAMES = 30  # 持續達標幀數（防呆：約 1 秒@30fps）


def similarity_to_score(similarity: float) -> int:
    """
    將餘弦相似度線性映射為 0~100 的百分制遊戲分數。

    Args:
        similarity: 餘弦相似度值，範圍 [-1.0, 1.0]

    Returns:
        百分制分數 (int)，範圍 [0, 100]
    """
    score = (similarity + 1.0) / 2.0 * 100.0
    return int(np.clip(score, 0, 100))


def draw_score_overlay(frame: np.ndarray, score: int, target_name: str,
                       pass_counter: int) -> None:
    """
    在畫面上渲染分數、目標姿勢名稱與通關狀態覆蓋層。

    Args:
        frame: BGR 影像幀（直接修改，無返回值）
        score: 當前百分制分數 (0~100)
        target_name: 當前目標姿勢名稱
        pass_counter: 當前持續達標幀數
    """
    h, w = frame.shape[:2]

    # 半透明黑色資訊欄（左上角）
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 130), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

    # 目標姿勢名稱
    cv2.putText(frame, f"目標：{target_name}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    # 分數顯示（依分數高低變換顏色）
    score_color = (0, 255, 0) if score >= PASS_THRESHOLD else (0, 165, 255)
    cv2.putText(frame, f"分數：{score} / 100", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, score_color, 2)

    # 進度條
    bar_w = int((w - 40) * score / 100)
    cv2.rectangle(frame, (20, 95), (w - 20, 115), (80, 80, 80), -1)
    cv2.rectangle(frame, (20, 95), (20 + bar_w, 115), score_color, -1)

    # 通關倒數顯示
    if pass_counter > 0:
        remain = PASS_HOLD_FRAMES - pass_counter
        cv2.putText(frame, f"保持姿勢！還需 {remain} 幀", (20, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)


def extract_pose_from_image(image_path: str) -> np.ndarray | None:
    """
    從靜態圖片檔案提取正規化姿態特徵向量。

    Args:
        image_path: 圖片檔案路徑（支援 jpg, png 等格式）

    Returns:
        正規化後的 66 維特徵向量，若偵測失敗則返回 None
    """
    image_path = str(Path(image_path))
    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        print(f"[錯誤] 無法讀取圖片：{image_path}")
        return None

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2
    ) as pose:
        results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print(f"[警告] 圖片中未偵測到人體：{image_path}")
        return None

    return normalize_pose(results.pose_landmarks.landmark)


# ── 建立多姿勢特徵庫 ────────────────────────────────────
POSE_LIBRARY: dict[str, np.ndarray] = {}
pose_images = {
    "T字形": "poses/t_pose.jpg",
    "Y字形": "poses/y_pose.jpg",
    "單腳站立": "poses/balance_pose.jpg",
}

for pose_name, image_path in pose_images.items():
    feature_vector = extract_pose_from_image(image_path)
    if feature_vector is not None:
        POSE_LIBRARY[pose_name] = feature_vector
        print(f"✓ 已載入目標姿勢：{pose_name}（特徵維度：{feature_vector.shape}）")
    else:
        print(f"✗ 載入失敗：{pose_name}")

print(f"\n特徵庫建立完成，共 {len(POSE_LIBRARY)} 個姿勢。")


def process_image_pose(image_path: str):
    print("MediaPipe Pose 初始化完成！")

    image_bgr = cv2.imread(image_path)
    if image_bgr is None:
        raise FileNotFoundError(f"找不到 {image_path}，請放入正確的圖片路徑。")

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    with mp_pose.Pose(static_image_mode=True) as pose:
        results = pose.process(image_rgb)

        if results.pose_landmarks:
            print("✓ 成功偵測到人體關鍵點！")
            nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            print(f" 鼻子位置：x={nose.x:.3f}, y={nose.y:.3f}, z={nose.z:.3f}")
            print(f" 可見度：{nose.visibility:.3f}")

            normalized = normalize_pose(results.pose_landmarks.landmark)
            if normalized is not None:
                print(f" 正規化後特徵向量長度：{normalized.shape[0]}")
            else:
                print("✗ 正規化失敗：關鍵點可能無效或尺度過小。")
            return normalized
        else:
            print("✗ 未偵測到人體，請確認圖片內容。")
            return None


def run_pose_detection_loop():
    """
    開啟攝影機並進行即時姿態偵測的主迴圈（基礎版本）。
    按下 'q' 鍵退出。
    """
    cap = cv2.VideoCapture(0)  # 開啟預設攝影機（索引 0）

    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機，請確認設備連接狀態。")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                print("[警告] 無法讀取攝影機畫面，跳過此幀。")
                continue

            # ★ 水平翻轉：提供直覺的鏡像互動體驗
            frame = cv2.flip(frame, 1)

            # BGR → RGB 轉換後進行推論
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False  # 鎖定記憶體提升效能
            results = pose.process(frame_rgb)
            frame_rgb.flags.writeable = True

            # 將推論結果繪製回原始 BGR 畫面
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 255, 0), thickness=2, circle_radius=3
                    ),
                    connection_drawing_spec=mp_drawing.DrawingSpec(
                        color=(0, 128, 255), thickness=2
                    )
                )

            cv2.imshow("MediaPipe Pose Detection", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("使用者按下 q，結束偵測。")
                break

    cap.release()
    cv2.destroyAllWindows()


def run_pose_game(pose_library: dict[str, np.ndarray], target_pose_name: str) -> None:
    """
    執行全身肢體相似度辨識遊戲主迴圈。

    Args:
        pose_library: 目標姿勢特徵庫 dict[str, np.ndarray]
        target_pose_name: 本輪目標姿勢鍵值（需存在於 pose_library）
    """
    if target_pose_name not in pose_library:
        raise ValueError(f"目標姿勢 '{target_pose_name}' 不存在於特徵庫中。")

    target_vector = pose_library[target_pose_name]
    pass_counter = 0  # 持續達標幀數計數器
    game_cleared = False

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("無法開啟攝影機。")

    with mp_pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as pose_estimator:

        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                continue

            frame = cv2.flip(frame, 1)  # 水平鏡像翻轉

            # ── 姿態偵測 ──────────────────────────────────
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = pose_estimator.process(frame_rgb)
            frame_rgb.flags.writeable = True

            score = 0  # 預設分數

            if results.pose_landmarks:
                # 繪製骨架
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks,
                    mp_pose.POSE_CONNECTIONS
                )

                live_vector = normalize_pose(results.pose_landmarks.landmark)
                if live_vector is not None:
                    sim = cosine_similarity(target_vector, live_vector)
                    score = similarity_to_score(sim)

            # ── 狀態機：通關判定 ──────────────────
            if score >= PASS_THRESHOLD:
                pass_counter += 1
            else:
                pass_counter = 0

            if pass_counter >= PASS_HOLD_FRAMES:
                game_cleared = True

            # ── 渲染覆蓋層 ────────────────────────────────
            draw_score_overlay(frame, score, target_pose_name, pass_counter)
            if game_cleared:
                h, w = frame.shape[:2]
                cv2.putText(frame, "通關！請保持姿勢以完成挑戰。", (20, h - 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.imshow("Pose Game", frame)

            if cv2.waitKey(5) & 0xFF == ord('q'):
                print("使用者按下 q，結束遊戲。")
                break

    cap.release()
    cv2.destroyAllWindows()


class GameEngine:
    """
    遊戲引擎：管理狀態機、分數計算與 UI 渲染。
    整合 PoseEstimator 與 MathUtils 協調完整遊戲流程。
    """

    PASS_THRESHOLD = 80  # 通關分數閾值
    PASS_HOLD_FRAMES = 30  # 持續達標幀數
    SMA_WINDOW = 10  # 移動平均視窗大小

    def __init__(
        self,
        pose_library: dict,
        pose_estimator: "PoseEstimator"
    ) -> None:
        self.pose_library = pose_library
        self.estimator = pose_estimator
        self._score_buffer = deque()
        self._pass_counter = 0
        self._game_cleared = False

    def _render_ui(self, frame: np.ndarray, score: int,
                   target_name: str) -> None:
        """在畫面上渲染分數條、目標名稱與通關狀態。"""
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (360, 140), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

        cv2.putText(frame, f"Target: {target_name}",
                    (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2)

        color = (0, 255, 0) if score >= self.PASS_THRESHOLD else (0, 165, 255)
        cv2.putText(frame, f"Score: {score} / 100",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 2)

        bar_w = int((w - 40) * score / 100)
        cv2.rectangle(frame, (20, 100), (w - 20, 125), (60, 60, 60), -1)
        cv2.rectangle(frame, (20, 100), (20 + bar_w, 125), color, -1)

        if self._pass_counter > 0 and not self._game_cleared:
            remain = self.PASS_HOLD_FRAMES - self._pass_counter
            cv2.putText(frame, f"保持！還需 {remain} 幀",
                        (20, h - 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.9, (0, 255, 255), 2)

        if self._game_cleared:
            cv2.putText(frame, "LEVEL CLEAR!", (w // 2 - 180, h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 255, 0), 4)

    def run(self, target_pose_name: str) -> None:
        """
        執行遊戲主迴圈。

        Args:
            target_pose_name: 目標姿勢名稱（須存在於 pose_library）
        """
        if target_pose_name not in self.pose_library:
            raise ValueError(f"姿勢 '{target_pose_name}' 不在特徵庫中。")

        target_vector = self.pose_library[target_pose_name]
        self._score_buffer.clear()
        self._pass_counter = 0
        self._game_cleared = False

        while True:
            frame = self.estimator.read_frame()
            if frame is None:
                continue

            annotated_frame, landmarks = self.estimator.process_frame(frame)
            raw_score = 0

            if landmarks:
                live_vec = MathUtils.normalize_pose(landmarks.landmark)
                if live_vec is not None:
                    sim = MathUtils.cosine_similarity(target_vector, live_vec)
                    raw_score = MathUtils.similarity_to_score(sim)

            smoothed_score = int(MathUtils.moving_average_filter(
                self._score_buffer, raw_score, self.SMA_WINDOW
            ))

            if smoothed_score >= self.PASS_THRESHOLD:
                self._pass_counter += 1
            else:
                self._pass_counter = 0

            if self._pass_counter >= self.PASS_HOLD_FRAMES:
                self._game_cleared = True

            self._render_ui(annotated_frame, smoothed_score, target_pose_name)
            cv2.imshow("Pose Similarity Game", annotated_frame)

            key = cv2.waitKey(5) & 0xFF
            if key == ord('q') or (self._game_cleared and key != 255):
                break


if __name__ == "__main__":
    normalized_pose = process_image_pose("test_pose.jpg")

    # 使用示例：計算兩個正規化姿態向量之間的相似度
    vec_target = np.random.randn(66).astype(np.float32)
    vec_live = vec_target + np.random.randn(66).astype(np.float32) * 0.1

    sim = cosine_similarity(vec_target, vec_live)
    score = similarity_to_score(sim)
    print(f"餘弦相似度：{sim:.4f}")
    print(f"遊戲分數：{score} 分")

    # 若要開啟即時攝影機偵測，取消下方註解
    # run_pose_detection_loop()
    # run_pose_game(POSE_LIBRARY, "T字形")

    with PoseEstimator(camera_index=0, model_complexity=1) as estimator:
        engine = GameEngine(
            pose_library=POSE_LIBRARY,
            pose_estimator=estimator
        )
        engine.run(target_pose_name="T字形")
