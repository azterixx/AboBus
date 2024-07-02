import time

import numpy as np
import supervision as sv
from supervision.assets import VideoAssets, download_assets
from ultralytics import YOLO

model = YOLO("yolov8n.pt")
byte_tracker = sv.ByteTrack()
bounding_box_annotator = sv.BoundingBoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=4, text_scale=2)
trace_annotator = sv.TraceAnnotator(thickness=4)

START = sv.Point(0, 440)
END = sv.Point(1920, 440)

line_zone = sv.LineZone(start=START, end=END)
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4,
    text_thickness=4,
    text_scale=2,
)


def callback(frame: np.ndarray, index: int) -> np.ndarray:
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    detections = byte_tracker.update_with_detections(detections)
    value_key_dict = {value: key for key, value in model.model.names.items()}
    person_class_id = value_key_dict["person"]
    detections = detections[detections.class_id == person_class_id]  # type: ignore

    labels = [
        f"#{tracker_id} {model.model.names[class_id]} #{class_id} {confidence:0.2f}"  # type: ignore
        for confidence, class_id, tracker_id in zip(
            detections.confidence,  # type: ignore
            detections.class_id,  # type: ignore
            detections.tracker_id,  # type: ignore
        )
    ]

    annotated_frame = frame.copy()
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
        labels=labels,
    )
    detections.xyxy[:, 3] = (
        detections.xyxy[:, 1] + (detections.xyxy[:, 3] - detections.xyxy[:, 1]) / 4
    )
    annotated_frame = bounding_box_annotator.annotate(
        scene=annotated_frame,
        detections=detections,
    )
    line_zone.trigger(detections)

    return line_zone_annotator.annotate(
        annotated_frame,
        line_counter=line_zone,
    )


if __name__ == "__main__":
    start = time.time()
    SOURCE_VIDEO_PATH = "door2.mp4"
    TARGET_VIDEO_PATH = "processed_door2.mp4"
    sv.process_video(
        source_path=SOURCE_VIDEO_PATH,
        target_path=TARGET_VIDEO_PATH,
        callback=callback,
    )
    print(time.time() - start)
    # video_info = sv.VideoInfo.from_video_path("door.mp4")
    # print(video_info.resolution_wh)
