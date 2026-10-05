import cv2
import numpy as np

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert the color frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Canny edge detection
    edges = cv2.Canny(gray, 20, 100)

    # Convert the grayscale edge image to 3 channels
    edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Place the original frame and edge image side by side
    combined = np.hstack((frame, edges_color))

    cv2.imshow("Original Video | Canny Edges", combined)

    # Press q or Esc to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

cap.release()
cv2.destroyAllWindows()