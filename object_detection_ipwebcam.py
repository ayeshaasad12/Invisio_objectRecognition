# Import required libraries
import cv2  # For image and video processing
import numpy as np  # For numerical operations
import tensorflow as tf  # For running the TFLite model
import pyttsx3  # For text-to-speech voice feedback
import threading  # For running voice feedback in parallel
import time  # For FPS calculation and time-related functions

# ESP32 IP Camera Stream URL
IP_WEBCAM_STREAM = "http://192.168.192.86:81/stream"

# Class to load and handle YOLO TFLite model
class ObjectDetector:
    def __init__(self, model_path, label_path):
        # Load the TFLite model
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()  # Allocate memory for model tensors

        # Get model input and output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()
        self.input_shape = self.input_details[0]['shape']  # Model input shape (batch, height, width, channels)

        # Load label names from file
        with open(label_path, 'r') as f:
            self.labels = f.read().strip().split("\n")

    # Method to detect objects in a single frame
    def detect_objects(self, frame):
        # Resize frame to model input size and normalize pixels
        resized = cv2.resize(frame, (self.input_shape[1], self.input_shape[2]))
        input_data = np.expand_dims(resized, axis=0).astype(np.float32) / 255.0

        # Set input tensor and run inference
        self.interpreter.set_tensor(self.input_details[0]['index'], input_data)
        self.interpreter.invoke()

        # Get model output and process results
        output_data = self.interpreter.get_tensor(self.output_details[0]['index'])
        return self.process_output(output_data, frame.shape)

    # Method to process raw model output and convert it into readable detections
    def process_output(self, output_data, image_shape):
        height, width, _ = image_shape
        detections = []

        # Iterate over each detected object
        for det in output_data[0]:
            conf = det[4]  # Confidence score
            if conf > 0.5:  # Filter out low-confidence detections
                class_id = int(np.argmax(det[5:]))  # Get class with highest confidence
                label = self.labels[class_id]  # Get corresponding label
                x_center, y_center, w, h = det[:4]  # Bounding box info (center x/y and width/height)

                # Convert center coordinates and size to pixel positions
                x_center *= width
                y_center *= height
                w *= width
                h *= height
                x_min = int(x_center - w / 2)
                y_min = int(y_center - h / 2)
                x_max = int(x_center + w / 2)
                y_max = int(y_center + h / 2)

                # Append formatted detection to results
                detections.append({'label': label, 'confidence': float(conf), 'box': [x_min, y_min, x_max, y_max]})
        return detections

# Non-Maximum Suppression to remove overlapping boxes
def apply_nms(detections, iou_threshold=0.3):
    boxes = [d['box'] for d in detections]
    scores = [d['confidence'] for d in detections]

    # Apply OpenCV NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, 0.5, iou_threshold)

    # Return filtered detections
    if len(indices) == 0:
        return []
    indices = indices.flatten()
    return [detections[i] for i in indices]

# Class to handle voice feedback using text-to-speech
class VoiceFeedback:
    def __init__(self):
        self.engine = pyttsx3.init()  # Initialize TTS engine
        self.set_female_voice()  # Set a female voice
        self.engine.setProperty("rate", 125)  # Set speaking speed

    # Select a female voice if available
    def set_female_voice(self):
        voices = self.engine.getProperty('voices')
        for voice in voices:
            if 'female' in voice.name.lower() or 'zira' in voice.name.lower():
                self.engine.setProperty('voice', voice.id)
                break

    # Speak the message in a separate thread (non-blocking)
    def speak(self, message):
        threading.Thread(target=self._play, args=(message,), daemon=True).start()

    # Private method to play the voice message
    def _play(self, msg):
        print("Speaking:", msg)
        self.engine.say(msg)
        self.engine.runAndWait()

# Helper function to determine object position (left, right, front)
def get_position(box, image_shape):
    height, width, _ = image_shape
    x_min, _, x_max, _ = box
    center_x = (x_min + x_max) / 2

    # Divide frame into thirds for relative position
    if center_x < width / 3:
        return "to the left of you"
    elif center_x > 2 * width / 3:
        return "to the right of you"
    else:
        return "in front of you"

# Draw detections on the frame and announce new ones
def draw_detections(frame, detections, voice_feedback, announced_labels):
    for det in detections:
        label, conf, box = det['label'], det['confidence'], det['box']
        x_min, y_min, x_max, y_max = box

        # Get relative position and create message
        position = get_position(box, frame.shape)
        message = f"{label} {position}"

        # Draw bounding box and label
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.putText(frame, message, (x_min, y_max + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Speak message if not already announced
        if message not in announced_labels:
            voice_feedback.speak(message)
            announced_labels.add(message)

# Connect to the ESP32 IP camera stream
def fetch_frames():
    cap = cv2.VideoCapture(IP_WEBCAM_STREAM)
    if not cap.isOpened():
        print("Error: Cannot connect to ESP32 camera.")
        return None
    return cap

# Main object detection and voice feedback loop
def detection_loop(detector, voice_feedback):
    cap = fetch_frames()
    if cap is None:
        return
    announced = set()  # Track announced labels to avoid repetition

    while True:
        start = time.time()  # For FPS calculation

        ret, frame = cap.read()
        if not ret:
            continue

        # Detect objects in the current frame
        detections = detector.detect_objects(frame)

        # Apply NMS to remove duplicates
        detections = apply_nms(detections)

        # Draw and announce detections
        draw_detections(frame, detections, voice_feedback, announced)

        # Show FPS on frame
        fps = 1 / (time.time() - start)
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Object Detection", frame)

        # Exit if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Entry point of the script
def main():
    # Paths to model and labels
    model_path = r"C:\Users\kk\object_detection_app\assets\yolov5.tflite"
    label_path = r"C:\Users\kk\object_detection_app\assets\yolov5_labels.txt"

    # Initialize model and voice engine
    detector = ObjectDetector(model_path, label_path)
    voice = VoiceFeedback()

    print("Starting...")
    # Start real-time detection loop
    detection_loop(detector, voice)

# Run the main function
if __name__ == "__main__":
    main()
