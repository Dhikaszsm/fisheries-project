# 📸 Face Recognition Tutorial - Simple Version
# Tutorial untuk membaca muka user dari foto

import cv2
import numpy as np
import os
from datetime import datetime

class SimpleFaceDetector:
    """
    Simple Face Detection Tutorial
    Menggunakan OpenCV Haar Cascade untuk detect wajah
    """
    
    def __init__(self):
        """
        Initialize face detector
        """
        # Load pre-trained Haar Cascade model untuk face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Create folder untuk menyimpan hasil
        self.output_folder = 'face_results'
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)
            
        print("🎯 Simple Face Detector initialized!")
        print(f"📁 Output folder: {self.output_folder}")
    
    def detect_faces_from_image(self, image_path):
        """
        Detect faces dari file gambar
        
        Args:
            image_path (str): Path ke file gambar
            
        Returns:
            tuple: (original_image, faces_detected, face_locations)
        """
        print(f"\n🔍 Processing image: {image_path}")
        
        # Read image
        if not os.path.exists(image_path):
            print(f"❌ File not found: {image_path}")
            return None, 0, []
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image")
            return None, 0, []
        
        # Convert to grayscale (face detection works better on grayscale)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,    # How much the image size is reduced at each scale
            minNeighbors=5,     # How many neighbors each candidate rectangle should retain
            minSize=(30, 30),   # Minimum possible face size
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        print(f"👥 Found {len(faces)} face(s) in image")
        
        # Draw rectangles around faces
        result_image = image.copy()
        for i, (x, y, w, h) in enumerate(faces):
            # Draw rectangle around face
            cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
            # Add label
            label = f"Face {i+1}"
            cv2.putText(result_image, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Print face info
            print(f"   Face {i+1}: Position({x}, {y}), Size({w}x{h})")
        
        return result_image, len(faces), faces
    
    def extract_face_region(self, image_path, save_faces=True):
        """
        Extract dan simpan setiap wajah yang ditemukan
        
        Args:
            image_path (str): Path ke file gambar
            save_faces (bool): Simpan setiap wajah ke file terpisah
        """
        print(f"\n✂️  Extracting faces from: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image")
            return []
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        extracted_faces = []
        
        for i, (x, y, w, h) in enumerate(faces):
            # Extract face region
            face_region = image[y:y+h, x:x+w]
            extracted_faces.append(face_region)
            
            if save_faces:
                # Generate filename
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f"face_{i+1}_{timestamp}.jpg"
                filepath = os.path.join(self.output_folder, filename)
                
                # Save face
                cv2.imwrite(filepath, face_region)
                print(f"💾 Saved face {i+1}: {filename}")
        
        return extracted_faces
    
    def process_webcam(self, duration=10):
        """
        Real-time face detection dari webcam
        
        Args:
            duration (int): Durasi recording dalam detik
        """
        print(f"\n📹 Starting webcam face detection for {duration} seconds")
        print("Press 'q' to quit early")
        
        # Open webcam
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            print("❌ Could not open webcam")
            return
        
        start_time = datetime.now()
        face_count = 0
        
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                break
            
            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Detect faces
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
            
            # Draw rectangles
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame, 'Face Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Show frame
            cv2.imshow('Face Detection', frame)
            
            # Count unique faces (simple logic)
            if len(faces) > 0:
                face_count += 1
            
            # Check if time is up or user pressed 'q'
            elapsed = (datetime.now() - start_time).seconds
            if elapsed >= duration or cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        
        print(f"✅ Webcam session finished!")
        print(f"📊 Face detections: {face_count} frames")
    
    def analyze_image_quality(self, image_path):
        """
        Analyze image quality untuk face recognition
        
        Args:
            image_path (str): Path ke file gambar
        """
        print(f"\n🔍 Analyzing image quality: {image_path}")
        
        image = cv2.imread(image_path)
        if image is None:
            print("❌ Could not load image")
            return
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate image quality metrics
        height, width = gray.shape
        
        # Brightness
        brightness = np.mean(gray)
        
        # Contrast (standard deviation)
        contrast = np.std(gray)
        
        # Sharpness (Laplacian variance)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Face detection
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)
        
        print("📊 IMAGE QUALITY ANALYSIS")
        print("=" * 30)
        print(f"📏 Resolution: {width}x{height}")
        print(f"💡 Brightness: {brightness:.1f} (Good: 80-180)")
        print(f"🎨 Contrast: {contrast:.1f} (Good: >30)")  
        print(f"🔍 Sharpness: {laplacian_var:.1f} (Good: >100)")
        print(f"👥 Faces detected: {len(faces)}")
        
        # Quality assessment
        quality_score = 0
        recommendations = []
        
        if 80 <= brightness <= 180:
            quality_score += 25
            print("✅ Brightness: Good")
        else:
            recommendations.append("Adjust lighting" if brightness < 80 else "Reduce brightness")
            print(f"⚠️  Brightness: {'Too dark' if brightness < 80 else 'Too bright'}")
        
        if contrast > 30:
            quality_score += 25
            print("✅ Contrast: Good")
        else:
            recommendations.append("Improve contrast")
            print("⚠️  Contrast: Too low")
        
        if laplacian_var > 100:
            quality_score += 25
            print("✅ Sharpness: Good")
        else:
            recommendations.append("Reduce blur/improve focus")
            print("⚠️  Sharpness: Too blurry")
        
        if len(faces) == 1:
            quality_score += 25
            print("✅ Face detection: Perfect (1 face)")
        elif len(faces) > 1:
            recommendations.append("Ensure only one face in image")
            print(f"⚠️  Face detection: Multiple faces ({len(faces)})")
        else:
            recommendations.append("Ensure face is visible")
            print("❌ Face detection: No face detected")
        
        print(f"\n🎯 Overall Quality Score: {quality_score}/100")
        
        if recommendations:
            print("\n💡 Recommendations:")
            for i, rec in enumerate(recommendations, 1):
                print(f"   {i}. {rec}")
        else:
            print("\n🎉 Image quality is excellent for face recognition!")

def run_tutorial():
    """
    Main tutorial function
    """
    print("=" * 50)
    print("🎓 FACE RECOGNITION TUTORIAL")
    print("=" * 50)
    
    # Initialize detector
    detector = SimpleFaceDetector()
    
    while True:
        print("\n📋 CHOOSE AN OPTION:")
        print("1. 📸 Detect faces from image file")
        print("2. ✂️  Extract faces from image")
        print("3. 📹 Real-time webcam detection")
        print("4. 🔍 Analyze image quality")
        print("5. 🎓 Show tutorial info")
        print("6. ❌ Exit")
        
        choice = input("\nEnter choice (1-6): ").strip()
        
        if choice == '1':
            image_path = input("Enter image path: ").strip()
            if image_path:
                result_image, face_count, faces = detector.detect_faces_from_image(image_path)
                if result_image is not None:
                    # Save result
                    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                    output_path = os.path.join(detector.output_folder, f"detected_{timestamp}.jpg")
                    cv2.imwrite(output_path, result_image)
                    print(f"💾 Result saved: {output_path}")
        
        elif choice == '2':
            image_path = input("Enter image path: ").strip()
            if image_path:
                faces = detector.extract_face_region(image_path)
                print(f"✅ Extracted {len(faces)} faces")
        
        elif choice == '3':
            duration = input("Enter duration in seconds (default 10): ").strip()
            try:
                duration = int(duration) if duration else 10
                detector.process_webcam(duration)
            except ValueError:
                print("Invalid duration, using 10 seconds")
                detector.process_webcam(10)
        
        elif choice == '4':
            image_path = input("Enter image path: ").strip()
            if image_path:
                detector.analyze_image_quality(image_path)
        
        elif choice == '5':
            show_tutorial_info()
        
        elif choice == '6':
            print("👋 Tutorial selesai!")
            break
        
        else:
            print("❌ Invalid choice!")

def show_tutorial_info():
    """
    Show tutorial information
    """
    print("\n" + "=" * 50)
    print("📚 FACE RECOGNITION TUTORIAL INFO")
    print("=" * 50)
    print("""
🎯 WHAT THIS TUTORIAL DOES:
   • Detect faces in photos
   • Extract face regions  
   • Real-time webcam detection
   • Analyze image quality for face recognition

📋 REQUIREMENTS:
   • Python with OpenCV installed
   • Webcam (for real-time detection)
   • Image files (JPG, PNG, etc.)

💡 TIPS FOR BETTER RESULTS:
   • Use good lighting
   • Face should be clearly visible
   • Avoid blurry images
   • One face per image works best

🔧 HOW IT WORKS:
   1. Haar Cascade classifier detects faces
   2. OpenCV draws rectangles around detected faces
   3. Face regions can be extracted and saved
   4. Quality metrics help improve recognition

📁 OUTPUT:
   • Results saved in 'face_results' folder
   • Face images extracted individually
   • Quality analysis reports
""")

if __name__ == "__main__":
    # Pastikan OpenCV terinstall
    try:
        import cv2
        print("✅ OpenCV is available")
        run_tutorial()
    except ImportError:
        print("❌ OpenCV not installed!")
        print("Install dengan: pip install opencv-python")
