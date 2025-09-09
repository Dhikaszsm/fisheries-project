# Create sample image with face-like pattern for testing
import cv2
import numpy as np

def create_face_sample():
    """
    Create a sample image with face-like patterns for testing
    """
    print("Creating sample face image...")
    
    # Create image
    img = np.ones((400, 300, 3), dtype=np.uint8) * 240  # Light gray background
    
    # Draw face-like shape (circle)
    center = (150, 200)
    radius = 80
    cv2.circle(img, center, radius, (200, 180, 160), -1)  # Face color
    
    # Eyes (two smaller circles)
    eye1_center = (130, 180)
    eye2_center = (170, 180) 
    cv2.circle(img, eye1_center, 8, (50, 50, 50), -1)   # Left eye
    cv2.circle(img, eye2_center, 8, (50, 50, 50), -1)   # Right eye
    
    # Nose (small line)
    cv2.line(img, (150, 190), (150, 210), (150, 120, 100), 2)
    
    # Mouth (arc)
    cv2.ellipse(img, (150, 230), (20, 10), 0, 0, 180, (150, 100, 100), 2)
    
    # Add text
    cv2.putText(img, 'Sample Face', (100, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.putText(img, 'for Testing', (110, 350), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    
    # Save image
    filename = 'sample_face_test.jpg'
    cv2.imwrite(filename, img)
    print(f"Created: {filename}")
    
    return filename

def test_detection_on_sample():
    """
    Test face detection on the created sample
    """
    # Create sample image
    sample_file = create_face_sample()
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Load the sample image
    image = cv2.imread(sample_file)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Try different parameters for detection
    print("\nTesting face detection with different parameters...")
    
    params = [
        (1.05, 3),
        (1.1, 5),
        (1.2, 7),
        (1.3, 8)
    ]
    
    best_result = None
    max_faces = 0
    
    for scale, neighbors in params:
        faces = face_cascade.detectMultiScale(
            gray,
            scaleFactor=scale,
            minNeighbors=neighbors,
            minSize=(20, 20),
            maxSize=(200, 200)
        )
        
        print(f"Scale={scale}, Neighbors={neighbors}: Found {len(faces)} faces")
        
        if len(faces) > max_faces:
            max_faces = len(faces)
            best_result = (faces, scale, neighbors)
    
    # Use best result or default
    if best_result:
        faces, best_scale, best_neighbors = best_result
        print(f"\nBest result: {len(faces)} faces with scale={best_scale}, neighbors={best_neighbors}")
    else:
        faces = []
        print("No faces detected with any parameters")
    
    # Draw results
    result_image = image.copy()
    for i, (x, y, w, h) in enumerate(faces):
        cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(result_image, f'Face {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    
    # Save result
    result_filename = 'sample_detection_result.jpg'
    cv2.imwrite(result_filename, result_image)
    print(f"Detection result saved: {result_filename}")
    
    return len(faces) > 0

if __name__ == "__main__":
    print("=" * 50)
    print("SAMPLE FACE DETECTION DEMO")
    print("=" * 50)
    
    success = test_detection_on_sample()
    
    if success:
        print("\n✅ Face detection working!")
        print("Check 'sample_detection_result.jpg' for results")
    else:
        print("\n⚠️ No faces detected in sample")
        print("This is normal - OpenCV face detection works best on real photos")
    
    print("\n" + "=" * 50)
    print("NEXT STEPS:")
    print("1. Add your own photo (JPG/PNG) to this folder")
    print("2. Run: python test_photo.py")  
    print("3. The detector works better on real face photos")
    print("=" * 50)
