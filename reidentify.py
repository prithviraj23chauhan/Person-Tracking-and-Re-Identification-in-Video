import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from reid_utils import extract_features, load_features_db
from scipy.spatial.distance import cosine
from utils import get_similarity_color,config

def run_reidentification(video_path):
    model = YOLO(config["model_path"])
    deepsort = DeepSort(
        max_age=config["deepsort"]["max_age"],
        n_init=config["deepsort"]["n_init"],
        nn_budget=config["deepsort"]["nn_budget"]
    )
    cap = cv2.VideoCapture(video_path)
    features_db = load_features_db()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame,verbose=config["logging"]["verbose"])
        detections = results[0]
        raw_detections = []
        if detections.boxes is not None:
            for box in detections.boxes:
                if box.xyxy.shape[1] == 4:
                    xmin, ymin, xmax, ymax = box.xyxy[0].cpu().numpy()
                    confidence = box.conf.item()
                    class_id = int(box.cls.item())
                    width = xmax - xmin
                    height = ymax - ymin
                    bbox_xywh = [xmin, ymin, width, height]
                    raw_detections.append((bbox_xywh, confidence, "person"))
        trackers = deepsort.update_tracks(raw_detections, frame=frame)
        for track in trackers:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            track_id = track.track_id
            bbox = track.to_ltrb()
            features = extract_features(frame, bbox)
            best_similarity = None
            if features is not None and features_db:
                similarities = [1 - cosine(features, feat) for feat in features_db.values()]
                best_similarity = max(similarities)
                # Show best_id in the video
                # similarities = [(stored_id, 1 - cosine(features, feat)) for stored_id, feat in features_db.items()]
                # best_id, best_similarity = max(similarities, key=lambda x: x[1])
                # cv2.putText(frame, f"Match: {best_id} Sim: {best_similarity:.2f}", ...)

            if best_similarity is not None:
                color = get_similarity_color(best_similarity)
            else:
                color = (128, 128, 128)  # Gray if no similarity
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), color, 2)
            cv2.putText(frame, f"ID: {track_id} Sim: {best_similarity:.2f}" if best_similarity else f"ID: {track_id}", (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        cv2.imshow("ReID", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()