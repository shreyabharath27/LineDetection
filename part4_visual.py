import cv2
import numpy as np


def detect_canny(frame):
    """
    Convert the frame to grayscale and detect edges using Canny.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(
        gray,
        50,
        150
    )

    return edges


def detect_hough(frame, edges):
    """
    Detect line segments using the probabilistic Hough transform.
    """
    output = frame.copy()

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
                output,
                (int(x1), int(y1)),
                (int(x2), int(y2)),
                (0, 0, 255),
                3
            )

    return output


def detect_least_squares(frame, edges):
    """
    Fit one least-squares line to edge points in
    the left half and one line in the right half.
    """
    output = frame.copy()

    height, width = edges.shape

    regions = [
        (0, width // 2),
        (width // 2, width)
    ]

    for x_start, x_end in regions:
        region = edges[:, x_start:x_end]

        # Find coordinates of edge pixels
        y_points, x_points = np.where(region > 0)

        # Convert local x-coordinates to full-image coordinates
        x_points = x_points + x_start

        # Skip the region if there are not enough edge points
        if len(x_points) < 20:
            continue

        points = np.column_stack(
            (x_points, y_points)
        ).astype(np.float32)

        # Fit a line using least-squares distance
        line = cv2.fitLine(
            points,
            cv2.DIST_L2,
            0,
            0.01,
            0.01
        )

        vx, vy, x0, y0 = line.flatten()

        # Handle vertical lines
        if abs(vx) < 1e-6:
            x1 = int(x0)
            x2 = int(x0)
            y1 = 0
            y2 = height - 1

        else:
            x1 = x_start
            x2 = x_end - 1

            y1 = int(y0 + (x1 - x0) * vy / vx)
            y2 = int(y0 + (x2 - x0) * vy / vx)

        # Draw the fitted line in green
        cv2.line(
            output,
            (int(x1), int(y1)),
            (int(x2), int(y2)),
            (0, 255, 0),
            3
        )

    return output


def add_label(image, text):
    """
    Add a yellow label to an image.
    """
    cv2.putText(
        image,
        text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 255),
        2
    )

    return image


def compose_display(
    frame,
    edges,
    hough_result,
    least_squares_result
):
    """
    Create the 2x2 real-time visualization.
    """

    panel_width = 640
    panel_height = 360

    # Convert Canny edges to a 3-channel image
    canny_display = cv2.cvtColor(
        edges,
        cv2.COLOR_GRAY2BGR
    )

    # Create the combined result
    combined_result = cv2.addWeighted(
        hough_result,
        0.5,
        least_squares_result,
        0.5,
        0
    )

    # Resize the original and Canny images
    original_small = cv2.resize(
        frame,
        (panel_width // 2, panel_height)
    )

    canny_small = cv2.resize(
        canny_display,
        (panel_width // 2, panel_height)
    )

    # Add labels to the original/Canny panel
    add_label(original_small, "Original")
    add_label(canny_small, "Method A: Canny")

    # Place original and Canny side by side
    original_canny_panel = np.hstack(
        (
            original_small,
            canny_small
        )
    )

    # Resize the other panels
    hough_panel = cv2.resize(
        hough_result,
        (panel_width, panel_height)
    )

    least_squares_panel = cv2.resize(
        least_squares_result,
        (panel_width, panel_height)
    )

    combined_panel = cv2.resize(
        combined_result,
        (panel_width, panel_height)
    )

    # Add labels
    add_label(hough_panel, "Method B: Hough Lines")
    add_label(
        least_squares_panel,
        "Method C: Least Squares"
    )
    add_label(
        combined_panel,
        "Combined Result"
    )

    # Create the top row
    top_row = np.hstack(
        (
            original_canny_panel,
            hough_panel
        )
    )

    # Create the bottom row
    bottom_row = np.hstack(
        (
            least_squares_panel,
            combined_panel
        )
    )

    # Create the final 2x2 layout
    grid = np.vstack(
        (
            top_row,
            bottom_row
        )
    )

    return grid


def main():
    """
    Open the camera and run the real-time program.
    """

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        # Method A: Canny edge detection
        edges = detect_canny(frame)

        # Method B: Hough line detection
        hough_result = detect_hough(
            frame,
            edges
        )

        # Method C: Least-squares line fitting
        least_squares_result = detect_least_squares(
            frame,
            edges
        )

        # Create the side-by-side display
        display = compose_display(
            frame,
            edges,
            hough_result,
            least_squares_result
        )

        cv2.imshow(
            "Real-Time Detection Comparison",
            display
        )

        # Press q or Esc to quit
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q") or key == 27:
            break

    # Cleanly release the camera and close windows
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
