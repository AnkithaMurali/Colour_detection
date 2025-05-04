import cv2
import numpy as np

# Updated color classifier
def classify_color(h, s, v):
    if v < 50:
        return "Black"
    elif s < 50:
        return "White" if v > 200 else "Gray"
    if h < 10 or h > 170:
        return "Red"
    elif 10 <= h < 25:
        return "Orange"
    elif 25 <= h < 35:
        return "Yellow"
    elif 35 <= h < 85:
        return "Green"
    elif 85 <= h < 125:
        return "Blue"
    elif 125 <= h < 160:
        return "Purple"
    else:
        return "Unknown"

# Start camera
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    # Larger center ROI
    h, w, _ = frame.shape
    cx, cy = w // 2, h // 2
    size = 50  # bigger box now
    roi = frame[cy - size:cy + size, cx - size:cx + size]

    # Convert ROI to HSV and average color
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    avg_hsv = np.mean(hsv_roi.reshape(-1, 3), axis=0)
    h_val, s_val, v_val = avg_hsv.astype(int)

    detected_color = classify_color(h_val, s_val, v_val)

    # Display info
    cv2.putText(frame, f"Detected: {detected_color}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.rectangle(frame, (cx - size, cy - size), (cx + size, cy + size), (0, 255, 0), 2)

    # Show result
    cv2.imshow("Color Detection", frame)

    if cv2.waitKey(1) == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()
