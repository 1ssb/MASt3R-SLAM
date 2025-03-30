import cv2
import numpy as np
import yaml
import tkinter as tk

class ChessboardCalibrator:
    def __init__(self):
        # ----------------------------------------------------
        # Chessboard settings for A4 board:
        #  - 11x8 squares → 10x7 inner corners (vertices)
        #  - Each square is 25 mm
        # ----------------------------------------------------
        self.chessboard_size = (10, 7)  # inner corners: 10 per row, 7 per column
        self.square_size = 25.0         # size of each square in millimeters

        # Prepare object points: (0,0,0), (1,0,0), ... scaled by square size
        self.objp = np.zeros((self.chessboard_size[0] * self.chessboard_size[1], 3), np.float32)
        self.objp[:, :2] = np.mgrid[0:self.chessboard_size[0],
                                     0:self.chessboard_size[1]].T.reshape(-1, 2)
        self.objp *= self.square_size

        self.objpoints = []  # 3D points in real world space
        self.imgpoints = []  # 2D points in image plane

        self.capture_count = 0
        self.min_captures = 5  # Minimum images required for calibration

        # These will be updated each frame:
        self.current_frame = None
        self.current_gray = None
        self.current_found = False
        self.current_corners = None

        # Flag to signal finishing capture
        self.finished = False

    def on_capture(self):
        """Callback for the 'Capture' button."""
        if self.current_found:
            self.objpoints.append(self.objp)
            self.imgpoints.append(self.current_corners)
            self.capture_count += 1
            print(f"Captured frame #{self.capture_count}")
        else:
            print("Chessboard pattern not detected; cannot capture.")

    def on_finish(self):
        """Callback for the 'Finish' button."""
        if self.capture_count < self.min_captures:
            print(f"Need at least {self.min_captures} captures for calibration. "
                  f"Only {self.capture_count} captured.")
        else:
            self.finished = True

    def run(self):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("Error: Cannot open video capture")
            return

        window_name = "Chessboard Calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

        # Create a separate Tkinter window for the buttons
        root = tk.Tk()
        root.title("Calibration Controls")
        capture_button = tk.Button(root, text="Capture", command=self.on_capture)
        capture_button.pack(side="left", padx=10, pady=10)
        finish_button = tk.Button(root, text="Finish", command=self.on_finish)
        finish_button.pack(side="left", padx=10, pady=10)

        print("Focus the OpenCV video window and the Tkinter control window.")
        print("Press the buttons to capture images or finish.")
        print("You can also press ESC in the video window to exit.")

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            self.current_frame = frame.copy()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            self.current_gray = gray

            # Try to find the chessboard corners in the current frame
            found, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)
            self.current_found = found
            if found:
                # Refine corner positions for better accuracy
                criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                self.current_corners = corners
                cv2.drawChessboardCorners(frame, self.chessboard_size, corners, found)
            else:
                self.current_corners = None

            # Display instructions and capture status on the frame
            status_text = f"Captured: {self.capture_count} / {self.min_captures} images"
            cv2.putText(frame, status_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.putText(frame, "Use the Tkinter window to capture/finish", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

            cv2.imshow(window_name, frame)

            # Process Tkinter events
            root.update_idletasks()
            root.update()

            # Check for ESC key to exit
            key = cv2.waitKey(30) & 0xFF
            if key == 27:  # ESC key
                self.finished = True

            if self.finished:
                break

        cap.release()
        cv2.destroyAllWindows()
        root.destroy()

        if self.capture_count < self.min_captures:
            print("Not enough valid captures for calibration.")
            return

        # ----------------------------------------------------
        # Perform camera calibration using the captured images
        # ----------------------------------------------------
        ret_calib, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            self.objpoints, self.imgpoints, gray.shape[::-1], None, None
        )

        if not ret_calib:
            print("Calibration failed.")
            return

        print("Calibration succeeded.")
        print("Camera Matrix:")
        print(camera_matrix)
        print("Distortion Coefficients:")
        print(dist_coeffs)

        # ----------------------------------------------------
        # Write calibration data to a YAML file (OpenCV style)
        # ----------------------------------------------------
        calib_data = {
            "width": frame.shape[1],
            "height": frame.shape[0],
            "calibration": {
                "rows": 3,
                "cols": 3,
                "dt": "d",
                "data": camera_matrix.flatten().tolist()
            },
            "distortion": {
                "rows": 1,
                "cols": int(dist_coeffs.size),
                "dt": "d",
                "data": dist_coeffs.flatten().tolist()
            }
        }

        output_filename = "./calibration.yaml"
        with open(output_filename, "w") as f:
            yaml.dump(calib_data, f)

        print(f"Calibration data saved to {output_filename}")

if __name__ == '__main__':
    calibrator = ChessboardCalibrator()
    calibrator.run()
