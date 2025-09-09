# 📸 Face Recognition Tutorial

## 🎯 Tujuan
Tutorial ini mengajarkan cara membaca muka user dari foto menggunakan OpenCV.

## 📋 Prerequisites
```bash
pip install opencv-python numpy pillow
```

## 🚀 Quick Start

### 1. Test Webcam Face Detection
```bash
python simple_face_test.py
```
Output: File `test_face_HHMMSS.jpg` dengan kotak hijau di sekitar wajah

### 2. Interactive Demo
```bash
python face_demo.py
```
Pilihan:
- Option 1: Live webcam detection (tekan SPACE untuk capture)
- Option 2: Detect faces dalam file gambar

## 💡 Sample Code - Basic Face Detection

```python
import cv2

# Load face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Read image
image = cv2.imread('your_photo.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces
faces = face_cascade.detectMultiScale(gray, 1.1, 5)

# Draw rectangles
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

# Save result
cv2.imwrite('result.jpg', image)
print(f"Found {len(faces)} faces")
```

## 🔍 Advanced Example - Face Quality Check

```python
import cv2
import numpy as np

def check_face_quality(image_path):
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Basic quality metrics
    brightness = np.mean(gray)
    contrast = np.std(gray)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    print(f"Brightness: {brightness:.1f} (good: 80-180)")
    print(f"Contrast: {contrast:.1f} (good: >30)")
    print(f"Sharpness: {sharpness:.1f} (good: >100)")
    
    # Face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    print(f"Faces detected: {len(faces)}")

# Usage
check_face_quality('my_photo.jpg')
```

## 📸 Tips Foto Yang Baik

### ✅ GOOD
- Pencahayaan cukup terang
- Wajah menghadap depan  
- Background polos/tidak ramai
- Fokus tajam (tidak blur)
- Hanya 1 wajah di foto

### ❌ AVOID  
- Terlalu gelap/terang
- Wajah menyamping
- Background kompleks
- Foto blur/goyang
- Multiple faces (untuk enrollment)

## 🔧 Integration ke Fisheries System

### Step 1: Modify app.py
```python
import cv2

@app.route('/api/enroll-face-simple', methods=['POST'])
def enroll_face_simple():
    # Get uploaded image
    image_file = request.files.get('image')
    
    # Save temporarily  
    temp_path = f'temp_face_{datetime.now().strftime("%H%M%S")}.jpg'
    image_file.save(temp_path)
    
    # Detect face
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    image = cv2.imread(temp_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    # Clean up
    os.remove(temp_path)
    
    if len(faces) == 1:
        return jsonify({'success': True, 'message': 'Face detected successfully!'})
    elif len(faces) == 0:
        return jsonify({'success': False, 'message': 'No face detected'})
    else:
        return jsonify({'success': False, 'message': 'Multiple faces detected'})
```

### Step 2: Add HTML Form
```html
<!-- In face_enrollment.html -->
<form action="/api/enroll-face-simple" method="post" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <button type="submit">Upload & Detect Face</button>
</form>
```

## 📊 Testing Results

Setelah running `python simple_face_test.py`, akan ada file:
- `test_face_HHMMSS.jpg` - Screenshot webcam dengan face detection
- `face_demo.py` - Interactive demo untuk testing lebih lanjut

## 🚀 Next Steps

1. **Basic Detection**: Gunakan sample code di atas
2. **Quality Check**: Implement face quality validation
3. **Integration**: Add ke Flask app untuk login system
4. **Database Storage**: Store face data untuk recognition
5. **Real Recognition**: Upgrade ke face_recognition library (butuh dlib)

## 🔗 Files Created

- `simple_face_test.py` - Quick test script
- `face_demo.py` - Interactive demo
- `test_face_HHMMSS.jpg` - Sample output
- `FACE_TUTORIAL.md` - This guide

## ⚡ Quick Commands

```bash
# Test face detection
python simple_face_test.py

# Interactive demo  
python face_demo.py

# Check if webcam works
python -c "import cv2; cap=cv2.VideoCapture(0); print('Webcam OK' if cap.isOpened() else 'No webcam'); cap.release()"

# Test with your photo
python -c "
import cv2
img = cv2.imread('your_photo.jpg')
if img is not None: 
    print('Image loaded OK')
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    print(f'Faces found: {len(faces)}')
else:
    print('Cannot load image')
"
```

## 🎉 Success!

Jika berhasil melihat kotak hijau di sekitar wajah di file output, berarti face detection sudah bekerja!
