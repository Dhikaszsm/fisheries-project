# 🧪 Simple Face Recognition Test
# Test script untuk coba face detection dengan mudah

import cv2
import os
import numpy as np
from datetime import datetime

def quick_face_test():
    """
    Quick test untuk face detection
    """
    print("[QUICK FACE DETECTION TEST]")
    print("=" * 40)
    
    # Check OpenCV
    print(f"[OK] OpenCV version: {cv2.__version__}")
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Test with webcam if available
    print("\n[TEST] Testing webcam...")
    cap = cv2.VideoCapture(0)
    
    if cap.isOpened():
        print("[OK] Webcam detected!")
        
        # Take one frame for testing
        ret, frame = cap.read()
        if ret:
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            print(f"👥 Faces detected in webcam: {len(faces)}")
            
            # Draw rectangles around faces
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Save test image
            test_filename = f"webcam_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(test_filename, frame)
            print(f"💾 Test image saved: {test_filename}")
            
        cap.release()
        
    else:
        print("❌ No webcam found")
    
    # Test with sample image if exists
    print("\n📸 Looking for test images...")
    
    # Common image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    current_dir = os.getcwd()
    
    test_images = []
    for file in os.listdir(current_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            test_images.append(file)
    
    if test_images:
        print(f"📁 Found {len(test_images)} images:")
        for img in test_images[:5]:  # Show first 5
            print(f"   - {img}")
        
        # Test first image
        test_img_path = test_images[0]
        print(f"\n🔍 Testing with: {test_img_path}")
        
        image = cv2.imread(test_img_path)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5)
            
            print(f"👥 Faces detected in image: {len(faces)}")
            
            # Draw rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Save result
            result_filename = f"face_detected_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(result_filename, image)
            print(f"💾 Result saved: {result_filename}")
            
    else:
        print("📁 No test images found in current directory")
        print("💡 Add some JPG/PNG files to test face detection")

def create_sample_instruction():
    """
    Create instruction file for using the face recognition
    """
    instruction = """
🎓 FACE RECOGNITION TUTORIAL - Quick Start Guide

📋 STEP 1: INSTALL REQUIREMENTS
   pip install opencv-python numpy

📋 STEP 2: RUN TUTORIAL
   python face_tutorial.py

📋 STEP 3: TEST OPTIONS
   1. 📸 Image Detection - Upload foto dan detect wajah
   2. ✂️  Face Extraction - Extract wajah dari foto  
   3. 📹 Webcam Live - Real-time detection
   4. 🔍 Quality Check - Analyze foto quality

🎯 DEMO WORKFLOW:
   1. Ambil foto dengan HP/camera
   2. Copy foto ke folder project
   3. Run: python face_tutorial.py
   4. Choose option 1 (Image Detection)
   5. Enter path foto (contoh: myfoto.jpg)
   6. Lihat hasil di folder 'face_results'

💡 TIPS FOTO YANG BAIK:
   ✅ Pencahayaan cukup (tidak terlalu gelap/terang)
   ✅ Wajah menghadap camera
   ✅ Tidak blur/goyang
   ✅ Background tidak terlalu ramai
   ✅ Hanya 1 wajah di foto (untuk enrollment)

📁 OUTPUT FILES:
   • detected_YYYYMMDD_HHMMSS.jpg (foto dengan kotak hijau di wajah)
   • face_1_YYYYMMDD_HHMMSS.jpg (extract wajah saja)

🔧 TROUBLESHOOTING:
   • Error "No module named cv2" → pip install opencv-python
   • "Could not load image" → Check file path & format
   • "No face detected" → Improve lighting/photo quality

🚀 NEXT STEPS:
   Setelah berhasil detect wajah, bisa integrate ke:
   • Login system (app.py)
   • Attendance system
   • Access control
"""
    
    with open('FACE_RECOGNITION_GUIDE.txt', 'w', encoding='utf-8') as f:
        f.write(instruction)
    
    print("📄 Created: FACE_RECOGNITION_GUIDE.txt")

if __name__ == "__main__":
    print("[TEST] Starting Face Recognition Test...")
    
    # Run quick test
    quick_face_test()
    
    # Create instruction file
    create_sample_instruction()
    
    print("\n" + "=" * 50)
    print("[NEXT STEPS]")
    print("1. Run: python face_tutorial.py (for full tutorial)")
    print("2. Read: FACE_RECOGNITION_GUIDE.txt (for instructions)")
    print("3. Add photos to this folder for testing")
    print("=" * 50)
