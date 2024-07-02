import math

import cv2
from ultralytics import YOLO
from ultralytics.utils.plotting import Annotator

# Initialize model and video capture
model = YOLO("yolov8n.pt")
video_path = r"C:\Users\a-kst\PycharmProjects\AboBus\dd.mp4"
cap = cv2.VideoCapture(video_path)


# Get video properties
w, h, fps = (
    int(cap.get(x))
    for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
)

# Initialize video writer
output_path = "visioneye-distance-calculation.avi"
fourcc = cv2.VideoWriter_fourcc(*"MJPG")
out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
print(f"Video dimensions: {w}x{h}")

# Define parameters
center_point = (0, h)
pixel_per_meter = 10
door_coords = (275, 0, 350, 320)  # x1, y1, width, height
txt_color, txt_background, bbox_clr = (0, 0, 0), (255, 255, 255), (255, 0, 255)

# Process video frames
while True:
    ret, frame = cap.read()
    frame_id = int(round(cap.get(1)))
    if not ret:
        print(
            "Video frame is empty or video processing has been successfully completed."
        )
        break

    if frame_id % 3 != 0:
        continue

    annotator = Annotator(frame, line_width=2)
    door_x1, door_y1, door_width, door_height = door_coords

    # Draw door frame rectangle
    cv2.rectangle(
        frame,
        (door_x1, door_y1),
        (door_x1 + door_width, door_y1 + door_height),
        txt_background,
        1,
    )

    # Perform object tracking
    results = model.track(frame, persist=True, classes=[0], conf=0.7)
    boxes = results[0].boxes.xyxy.cpu()

    if results[0].boxes.id is not None:
        track_ids = results[0].boxes.id.int().cpu().tolist()

        for box, track_id in zip(boxes, track_ids):
            annotator.box_label(box, label=str(track_id), color=bbox_clr)

            # Calculate bounding box centroid
            x1, y1 = int((box[0] + box[2]) // 2), int((box[1] + box[3]) // 2)

            # Calculate distance to center point
            distance = (
                math.sqrt((x1 - center_point[0]) ** 2 + (y1 - center_point[1]) ** 2)
                / pixel_per_meter
            )

            # Annotate distance on frame
            text = f"Distance: {distance:.2f} m"
            text_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
            cv2.rectangle(
                frame,
                (x1, y1 - text_size[1] - 10),
                (x1 + text_size[0] + 10, y1),
                txt_background,
                -1,
            )
            cv2.putText(
                frame, text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 1.2, txt_color, 3
            )

    # Write frame to output video
    out.write(frame)
    cv2.imshow("visioneye-distance-calculation", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release resources
out.release()
cap.release()
cv2.destroyAllWindows()
