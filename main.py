import cv2
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
import os

# Set the path to the directory containing the model files
model_dir = "C:\\journal\\Data mining\\final my Datamining project\\aaa\\Models"  # Replace this with the actual directory path

# Function to detect faces and highlight them
def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight, frameWidth = frameOpencvDnn.shape[:2]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
    return frameOpencvDnn, faceBoxes

# Function to process the selected image
def process_image():
    filepath = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg;*.png;*.jpeg")])
    if not filepath:
        return

    frame = cv2.imread(filepath)
    if frame is None:
        messagebox.showerror("Error", "Could not read the image.")
        return

    resultImg, faceBoxes = highlightFace(faceNet, frame)
    if not faceBoxes:
        messagebox.showinfo("Result", "No face detected.")
    else:
        for faceBox in faceBoxes:
            face = frame[
                max(0, faceBox[1] - 20):min(faceBox[3] + 20, frame.shape[0] - 1),
                max(0, faceBox[0] - 20):min(faceBox[2] + 20, frame.shape[1] - 1)
            ]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)

            # Predict gender
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            # Predict age
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            age_numeric = int(age[1:-1].split('-')[0])

            # Set box color based on age
            box_color = (0, 255, 0) if age_numeric >= 18 else (0, 0, 255)

            cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), box_color, 2)
            cv2.putText(resultImg, f"{gender}, {age}", (faceBox[0], faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA)

        cv2.imshow("Processed Image", resultImg)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Function to process the live webcam feed
def process_webcam():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not access the webcam.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            messagebox.showerror("Error", "Failed to capture video.")
            break

        resultImg, faceBoxes = highlightFace(faceNet, frame)
        for faceBox in faceBoxes:
            face = frame[
                max(0, faceBox[1] - 20):min(faceBox[3] + 20, frame.shape[0] - 1),
                max(0, faceBox[0] - 20):min(faceBox[2] + 20, frame.shape[1] - 1)
            ]

            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), [124.96, 115.97, 106.13], swapRB=True, crop=False)

            # Predict gender
            genderNet.setInput(blob)
            genderPreds = genderNet.forward()
            gender = genderList[genderPreds[0].argmax()]

            # Predict age
            ageNet.setInput(blob)
            agePreds = ageNet.forward()
            age = ageList[agePreds[0].argmax()]
            age_numeric = int(age[1:-1].split('-')[0])

            # Set box color based on age
            box_color = (0, 255, 0) if age_numeric >= 18 else (0, 0, 255)

            cv2.rectangle(resultImg, (faceBox[0], faceBox[1]), (faceBox[2], faceBox[3]), box_color, 2)
            cv2.putText(resultImg, f"{gender}, {age}", (faceBox[0], faceBox[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, box_color, 2, cv2.LINE_AA)

        cv2.imshow("Webcam - Press 'q' to Exit", resultImg)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Exit the application
def exit_application():
    root.destroy()

# Model files with full paths
faceProto = os.path.join(model_dir, "opencv_face_detector.pbtxt")
faceModel = os.path.join(model_dir, "opencv_face_detector_uint8.pb")
ageProto = os.path.join(model_dir, "age_deploy.prototxt")
ageModel = os.path.join(model_dir, "age_net.caffemodel")
genderProto = os.path.join(model_dir, "gender_deploy.prototxt")
genderModel = os.path.join(model_dir, "gender_net.caffemodel")

# Constants
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-5)', '(6-10)', '(11-15)', '(15-18)', '(19-25)', '(26-35)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Load models
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# GUI setup
root = tk.Tk()
root.title("Age and Gender Detection")
root.geometry("600x400")
root.configure(bg="#f0f8ff")

# Styling for buttons
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 14, "bold"), padding=10)
style.map("TButton", background=[("active", "#4682b4")], foreground=[("active", "white")])

# Heading
heading = tk.Label(root, text="Age and Gender Detection", font=("Helvetica", 24, "bold"), bg="#4682b4", fg="white")
heading.pack(pady=20)

# Buttons
image_button = ttk.Button(root, text="Process Image", command=process_image)
image_button.pack(pady=10)

webcam_button = ttk.Button(root, text="Use Webcam", command=process_webcam)
webcam_button.pack(pady=10)

exit_button = ttk.Button(root, text="Exit", command=exit_application)
exit_button.pack(pady=10)

# Run the application
root.mainloop()
