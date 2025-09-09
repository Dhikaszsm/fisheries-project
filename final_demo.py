# Final Face Recognition Demo
import cv2
import os
from datetime import datetime

def run_complete_demo():
    """
    Complete demo of face detection capabilities
    """
    print("=" * 50)
    print("FACE DETECTION COMPLETE DEMO")
    print("=" * 50)
    
    # Check OpenCV
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Initialize face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Test 1: Webcam availability
    print("\n[TEST 1] Webcam Test")
    print("-" * 30)
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        print("[OK] Webcam available")
        ret, frame = cap.read()
        if ret:
            print(f"[OK] Frame captured: {frame.shape}")
            
            # Quick face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            print(f"[INFO] Faces in current frame: {len(faces)}")
            
            # Save webcam test
            cv2.imwrite('webcam_test.jpg', frame)
            print("[SAVED] webcam_test.jpg")
        cap.release()
    else:
        print("[ERROR] No webcam available")
    
    # Test 2: Image files detection
    print("\n[TEST 2] Image Files Test")
    print("-" * 30)
    
    image_files = []
    for file in os.listdir('.'):
        if file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_files.append(file)
    
    print(f"[INFO] Found {len(image_files)} image files")
    
    results = []
    for img_file in image_files[:5]:  # Test first 5 images
        image = cv2.imread(img_file)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            results.append((img_file, len(faces)))
            print(f"[DETECT] {img_file}: {len(faces)} faces")
    
    # Test 3: Face Detection Parameters
    print("\n[TEST 3] Detection Parameters")
    print("-" * 30)
    
    if image_files:
        test_image = image_files[0]
        image = cv2.imread(test_image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Test different parameters
        params = [
            (1.05, 3, 30),
            (1.1, 5, 30),
            (1.3, 3, 50),
            (1.1, 8, 20)
        ]
        
        best_count = 0
        for scale, neighbors, min_size in params:
            faces = face_cascade.detectMultiScale(
                gray, 
                scaleFactor=scale, 
                minNeighbors=neighbors,
                minSize=(min_size, min_size)
            )
            print(f"[PARAM] Scale:{scale}, Neighbors:{neighbors}, MinSize:{min_size} -> {len(faces)} faces")
            if len(faces) > best_count:
                best_count = len(faces)
    
    # Test 4: Create Demo Result
    print("\n[TEST 4] Creating Demo Results")  
    print("-" * 30)
    
    if image_files:
        # Process first image with best detection
        demo_image = image_files[0]
        image = cv2.imread(demo_image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use optimal parameters
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30, 30))
        
        # Draw detection results
        result_image = image.copy()
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 3)
            
            # Add label
            label = f'FACE {i+1}'
            cv2.putText(result_image, label, (x, y-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add info text
        info_text = f'DETECTED: {len(faces)} FACES'
        cv2.putText(result_image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        timestamp = datetime.now().strftime('%H%M%S')
        demo_filename = f'DEMO_RESULT_{timestamp}.jpg'
        cv2.imwrite(demo_filename, result_image)
        print(f"[CREATED] {demo_filename}")
    
    # Summary
    print("\n" + "=" * 50)
    print("DEMO SUMMARY")
    print("=" * 50)
    print(f"Images tested: {len(image_files)}")
    print(f"Total detections: {sum([r[1] for r in results])}")
    
    if results:
        print("Results:")
        for filename, face_count in results:
            status = "[OK]" if face_count > 0 else "[NO FACE]"
            print(f"  {status} {filename}: {face_count} faces")
    
    print(f"\nOutput files created:")
    output_files = ['webcam_test.jpg'] + [f for f in os.listdir('.') if f.startswith('DEMO_RESULT_')]
    for file in output_files:
        if os.path.exists(file):
            print(f"  - {file}")
    
    print("\n[SUCCESS] Face detection demo completed!")
    print("Check the output files to see detection results.")

if __name__ == "__main__":
    try:
        run_complete_demo()
    except Exception as e:
        print(f"[ERROR] Demo failed: {e}")
        print("Make sure OpenCV is properly installed: pip install opencv-python")
