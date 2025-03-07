# Install necessary libraries (run this in your terminal, not in the script):
# pip install ultralytics tensorflow opencv-python-headless numpy

# Import required libraries
import cv2
from ultralytics import YOLO
import numpy as np

# Load the YOLOv8 model
model = YOLO('yolov8n.pt')  # Nano model for faster processing

# Open the webcam (0 is typically the default camera on MacBook)
cap = cv2.VideoCapture(0)

# Check if the webcam opened successfully
if not cap.isOpened():
    print("Error: Could not open the camera. Check if it's connected or accessible.")
    exit()

# Get video properties
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Verify video properties (for debugging)
print(f"Webcam resolution: {width}x{height}, FPS: {fps}")
if width == 0 or height == 0:
    print("Error: Invalid resolution. Webcam may not be functioning correctly.")
    cap.release()
    exit()

# Define the codec and create VideoWriter object
output_path = 'output_video.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# Check if VideoWriter initialized correctly
if not out.isOpened():
    print("Error: Could not initialize video writer.")
    cap.release()
    exit()

# Process the video frame by frame
while cap.isOpened():
    ret, frame = cap.read()
    if ret:
        # Convert the frame to RGB (YOLOv8 expects RGB format)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Perform object detection
        results = model.predict(rgb_frame)

        # Render the results (annotated frame with bounding boxes and labels)
        annotated_frame = results[0].plot()

        # Convert back to BGR for OpenCV
        bgr_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR)

        # Write the frame to the output video
        out.write(bgr_frame)

        # Display the frame in a window
        cv2.imshow('YOLOv8 Detection', bgr_frame)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        print("Error: Failed to capture frame from webcam.")
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()

print(f"Processing complete. Video saved as '{output_path}'.")