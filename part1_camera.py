import cv2

# Open the default camera
cap = cv2.VideoCapture(0)

# Check whether the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    # Read one frame from the camera
    ret, frame = cap.read()

    # Stop if a frame could not be read
    if not ret:
        print("Error: Could not read frame.")
        break

    # Display the live camera frame
    cv2.imshow("Live Camera Feed", frame)

    # Press q or Esc to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

# Release the camera
cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()