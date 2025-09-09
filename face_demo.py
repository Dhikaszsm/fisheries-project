# Face Recognition Demo - Simple Version

import cv2
from datetime import datetime

def detect_face_from_webcam():
    """Detect face from webcam and save image"""
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open webcam")
        return
    
    print("Press SPACE to capture, ESC to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(50, 50))
        
        # Draw rectangles around faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, f'Face {len(faces)}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Show frame
        cv2.imshow('Face Detection Demo', frame)
        
        # Handle key press
        key = cv2.waitKey(1) & 0xFF
        if key == 32:  # Space key
            if len(faces) > 0:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'captured_face_{timestamp}.jpg'
                cv2.imwrite(filename, frame)
                print(f"Captured: {filename}")
            else:
                print("No face detected!")
        elif key == 27:  # ESC key
            break
    
    cap.release()
    cv2.destroyAllWindows()

def detect_face_from_image(image_path):
    """Detect face from image file"""
    
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Cannot load image: {image_path}")
        return
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    print(f"Detected {len(faces)} faces in {image_path}")
    
    # Draw rectangles
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f'Face {i+1}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Save result
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_filename = f'detected_{timestamp}.jpg'
    cv2.imwrite(output_filename, image)
    print(f"Result saved: {output_filename}")

if __name__ == "__main__":
    print("FACE DETECTION DEMO")
    print("Choose option:")
    print("1. Webcam detection")
    print("2. Image file detection")
    
    choice = input("Enter choice (1 or 2): ")
    
    if choice == "1":
        detect_face_from_webcam()
    elif choice == "2":
        image_path = input("Enter image path: ")
        detect_face_from_image(image_path)
    else:
        print("Invalid choice")
