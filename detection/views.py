# Import necessary modules and libraries
from django.shortcuts import render
from django.http import JsonResponse
import cv2  # OpenCV for image processing
import numpy as np  # NumPy for array manipulation
import base64  # For base64 encoding and decoding
import os  # For file path manipulation
from django.conf import settings  # For accessing Django settings

# Define paths to YOLO weights and configuration files
YOLO_WEIGHTS = 'yolo/yolov3.weights'
YOLO_CONFIG = 'yolo/yolov3.cfg'
YOLO_NAMES = 'yolo/coco.names'

# Load YOLO model with weights and configuration files
net = cv2.dnn.readNet(YOLO_WEIGHTS, YOLO_CONFIG)

# Load class names from coco.names file
with open(YOLO_NAMES, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Function for displaying the index page
def index(request):
    # Render the index.html template when accessing the home page
    return render(request, 'detection/index.html')

# Function for object detection
def detect(request):
    if request.method == 'POST':
        # Receive image data sent from the form
        img_data = request.POST.get('image')
        img_data = img_data.split(',')[1]  # Separate base64 data from other data
        img_data = base64.b64decode(img_data)  # Decode base64 data to bytes
        np_arr = np.frombuffer(img_data, np.uint8)  # Create NumPy array from bytes
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)  # Decode image from NumPy array

        # Perform object detection
        blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)  # Convert image to blob
        net.setInput(blob)  # Set blob as input to the model
        outs = net.forward(net.getUnconnectedOutLayersNames())  # Perform forward pass to compute results

        class_ids = []
        confidences = []
        boxes = []

        # Loop through the detection results
        for out in outs:
            for detection in out:
                scores = detection[5:]  # Confidence scores of detected objects (green numbers)
                class_id = np.argmax(scores)  # Index of the class with the highest score
                confidence = scores[class_id]  # Highest confidence score
                if confidence > 0.5:  # Detect objects with confidence greater than 50%
                    center_x = int(detection[0] * img.shape[1])  # x-coordinate of object center
                    center_y = int(detection[1] * img.shape[0])  # y-coordinate of object center
                    w = int(detection[2] * img.shape[1])  # Width of the object
                    h = int(detection[3] * img.shape[0])  # Height of the object
                    x = int(center_x - w / 2)  # Calculate x-coordinate of top-left corner
                    y = int(center_y - h / 2)  # Calculate y-coordinate of top-left corner
                    boxes.append([x, y, w, h])  # Store object bounding box coordinates
                    confidences.append(float(confidence))  # Store confidence score
                    class_ids.append(class_id)  # Store class index

        # Remove overlapping boxes using Non-Maximum Suppression (NMS)
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

        detected_objects = []

        if len(indexes) > 0: 
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])  # Name of the detected object class
                confidence = confidences[i]  # Confidence score of the detected object
                color = (0, 255, 0)  # Set color of the bounding box (green)
                # Draw bounding box on the image
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                # Put class name and confidence score on the bounding box
                cv2.putText(img, f'{label} {confidence:.2f}', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                detected_objects.append(label)  # Add name of the detected object class to the list

        # Encode the image to base64
        _, buffer = cv2.imencode('.jpg', img)
        encoded_image = base64.b64encode(buffer).decode('utf-8')

        # Send the results back to the user in JSON format
        return JsonResponse({'detected_objects': detected_objects, 'image': encoded_image})
    
    # If the request method is not POST, return an error message
    return JsonResponse({'error': 'Invalid request method'}, status=400)
