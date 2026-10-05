import cv2
import numpy as np


def draw_least_squares_lines(frame, edges):
    """
    Fit one least-squares line to edge points in the
    left half and one line to edge points in the right half.
    """

    output = frame.copy()
    height, width = edges.shape

    # Define two image regions
    regions = [
        (0, width // 2),
        (width // 2, width)
    ]

    for x_start, x_end in regions:
        # Get edge-pixel coordinates in this region
        region = edges[:, x_start:x_end]
        y_points, x_points = np.where(region > 0)

        # Convert x-coordinates back to the full image
        x_points = x_points + x_start

        # Avoid fitting when there are too few points
        if len(x_points) < 20:
            continue

        # Limit the number of points for speed
        if len(x_points) > 3000:
            indices = np.random.choice(len(x_points), 3000, replace=False)
            x_points = x_points[indices]
            y_points = y_points[indices]

        # Least-squares line fitting using cv2.fitLine
        points = np.column_stack((x_points, y_points)).astype(np.float32)

        line = cv2.fitLine(
            points,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )

        vx, vy, x0, y0 = line.flatten()

        # Avoid division by zero for vertical lines
        if abs(vx) < 1e-6:
            x1 = x2 = int(x0)
            y1 = 0
            y2 = height - 1
        else:
            # Find the y-values at the left and right image boundaries
            x1 = x_start
            x2 = x_end - 1

            y1 = int(y0 + (x1 - x0) * vy / vx)
            y2 = int(y0 + (x2 - x0) * vy / vx)

        cv2.line(
            output,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )

    return output


# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Could not read frame.")
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Part 2: Canny edge detection
    edges = cv2.Canny(gray, 50, 150)

    # Method A: Canny edge pixels
    canny_display = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)

    # Method B: Probabilistic Hough transform
    hough_display = frame.copy()

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=50,
        minLineLength=50,
        maxLineGap=20
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.flatten()

            cv2.line(
                hough_display,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                3
            )

    # Method C: Least-squares line fitting
    least_squares_display = draw_least_squares_lines(frame, edges)

    # Add labels
    cv2.putText(
        canny_display,
        "Method A: Canny Edges",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        hough_display,
        "Method B: Hough Lines",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    cv2.putText(
        least_squares_display,
        "Method C: Least Squares",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    # Make the three displays smaller so they fit on the screen
    scale = 0.5

    canny_display = cv2.resize(
        canny_display,
        None,
        fx=scale,
        fy=scale
    )

    hough_display = cv2.resize(
        hough_display,
        None,
        fx=scale,
        fy=scale
    )

    least_squares_display = cv2.resize(
        least_squares_display,
        None,
        fx=scale,
        fy=scale
    )

    # Display all three approaches side by side
    combined = np.hstack(
        (
            canny_display,
            hough_display,
            least_squares_display
        )
    )

    cv2.imshow("Three Detection Approaches", combined)

    # Press q or Esc to quit
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q") or key == 27:
        break

# Clean up
cap.release()
cv2.destroyAllWindows()