# Quick test for face detection from photo
import cv2
import os
from datetime import datetime

def test_face_from_photo():
    """
    Test face detection from photo file
    """
    print("=" * 40)
    print("PHOTO FACE DETECTION TEST")  
    print("=" * 40)
    
    # Look for image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
    current_dir = os.getcwd()
    
    test_images = []
    for file in os.listdir(current_dir):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            test_images.append(file)
    
    if not test_images:
        print("No image files found!")
        print("Add some JPG/PNG files to this folder and try again")
        return
    
    print(f"Found {len(test_images)} images:")
    for i, img in enumerate(test_images):
        print(f"  {i+1}. {img}")
    
    # Test first image
    test_image = test_images[0] 
    print(f"\nTesting: {test_image}")
    
    # Load image
    image = cv2.imread(test_image)
    if image is None:
        print(f"Cannot load: {test_image}")
        return
    
    # Load face detector
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(
        gray, 
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )
    
    print(f"Faces detected: {len(faces)}")
    
    # Draw rectangles around faces
    for i, (x, y, w, h) in enumerate(faces):
        # Draw rectangle
        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 3)
        
        # Add label
        cv2.putText(image, f'Face {i+1}', (x, y-10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        print(f"  Face {i+1}: ({x},{y}) size {w}x{h}")
    
    # Save result
    timestamp = datetime.now().strftime('%H%M%S')
    result_filename = f'face_result_{timestamp}.jpg'
    cv2.imwrite(result_filename, image)
    
    print(f"Result saved: {result_filename}")
    print("\nOpen the result file to see detected faces!")

def create_sample_image():
    """
    Create a sample image for testing if no images found
    """
    print("\nCreating sample test image...")
    
    # Create simple image with text
    import numpy as np
    
    # Create white image
    img = np.ones((300, 400, 3), dtype=np.uint8) * 255
    
    # Add text
    cv2.putText(img, 'Sample Image', (100, 150), 
               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    cv2.putText(img, 'Add your photo here', (80, 200), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)
    cv2.putText(img, 'for face detection', (90, 230), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (128, 128, 128), 1)
    
    # Save
    cv2.imwrite('sample_image.jpg', img)
    print("Created: sample_image.jpg")

if __name__ == "__main__":
    test_face_from_photo()
    
    print("\n" + "=" * 40)
    print("INSTRUCTIONS:")
    print("1. Copy your photo (JPG/PNG) to this folder")
    print("2. Run: python test_photo.py")
    print("3. Check the 'face_result_HHMMSS.jpg' file")
    print("4. You should see green rectangles around faces")
    print("=" * 40)
