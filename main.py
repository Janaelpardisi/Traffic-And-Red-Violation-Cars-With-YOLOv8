import cv2
import numpy as np
from ultralytics import YOLO

# load YOLOv8n model
model = YOLO("yolov8n.pt")  

# open video
cap = cv2.VideoCapture(r"F:\traffic\tr.mp4")

# initialize width , height for video
frame_width, frame_height = 1020, 500

# region of signal_area
signal_area = (900, 92, 26, 22)  

# initialize violation_area
violation_area = [(324, 313), (283, 374), (854, 392), (864, 322)]

# function for initialize color of signal
def get_signal_color(roi):
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # range of red color
    lower_red1 = np.array([0, 50, 100])
    upper_red1 = np.array([10, 255, 255])
    lower_red2 = np.array([160, 50, 100])
    upper_red2 = np.array([179, 255, 255])
    red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    red_mask = cv2.bitwise_or(red_mask1, red_mask2)

    # range of green color
    lower_green = np.array([40, 50, 100])
    upper_green = np.array([90, 255, 255])
    green_mask = cv2.inRange(hsv, lower_green, upper_green)

    if cv2.countNonZero(red_mask) > 30:
        return "RED"
    elif cv2.countNonZero(green_mask) > 30:
        return "GREEN"
    else:
        return "UNKNOWN"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.resize(frame, (frame_width, frame_height))

    # extract region_signal and analysis color
    x, y, w, h = signal_area
    roi = frame[y:y+h, x:x+w]
    signal_color = get_signal_color(roi)

    # display signal
    signal_color_bgr = (0, 0, 255) if signal_color == "RED" else (0, 255, 0) if signal_color == "GREEN" else (128, 128, 128)
    cv2.rectangle(frame, (x, y), (x + w, y + h), signal_color_bgr, 2)
    cv2.putText(frame, signal_color, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, signal_color_bgr, 2)

    # display and draw violation of area
    cv2.polylines(frame, [np.array(violation_area, np.int32)], isClosed=True, color=(255, 30, 0), thickness=2)

    # detect cars
    results = model(frame, stream=True)
    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            name = model.names[cls_id]

            if name in ["car", "truck", "bus", "motorbike"]:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cx = (x1 + x2) // 2
                cy = (y1 + y2) // 2

                inside = cv2.pointPolygonTest(np.array(violation_area, np.int32), (cx, cy), False)

                if inside >= 0 and signal_color == "RED":
                    # violation
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, "Violation Car", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
                else:
                    # crossing without violation
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
                    cv2.circle(frame, (cx, cy), 5, (0, 255, 0), -1)

    cv2.imshow("Traffic Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()