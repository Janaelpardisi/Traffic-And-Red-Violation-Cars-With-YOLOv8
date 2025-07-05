🚦 Traffic Violation Detection using YOLOv8

This project is a real-time traffic violation detection system that automatically detects vehicles running red lights using computer vision and the YOLOv8 object detection model.

It monitors traffic scenes, detects the color of traffic lights (Red / Green), and identifies if any vehicle crosses the violation area while the signal is red. Such vehicles are marked as violators in the video feed.

🔍 Features

✅ YOLOv8-based vehicle detection (cars, trucks, buses, motorbikes).

🔴 Traffic light color detection using HSV color space.

🟥 Violation zone monitoring via polygon region.

🚗 Real-time tracking and marking of violators.

💡 Simple, efficient, and easily customizable for different videos or intersections.


📌 Technologies Used

Python

OpenCV

NumPy

Ultralytics YOLOv8


📽 How It Works

1. Detect traffic signal color using HSV thresholding.


2. Detect vehicles using YOLOv8.


3. Check if a vehicle is in the violation area while the signal is RED.


4. If yes → Mark as Violation Car with red box and label.





🛠 Use Cases

Smart traffic surveillance systems

Automatic fine generation

Road safety monitoring

AI-powered city solutions

