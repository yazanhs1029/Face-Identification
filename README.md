# Face Identification using CV2

A **real-time face recognition project** using **OpenCV (cv2)**, **TensorFlow**, and **face_recognition**.  
Detects faces from your webcam and identifies them based on a dataset of images.

---

## 📁 Dataset Setup

1. Create a folder for your people dataset (default: `people`).
2. Place images of each person in the folder.  
   - Supported formats: `.jpg`, `.jpeg`, `.png`
3. Update the `people_dataset` variable in the code with your folder path:
```python
people_dataset = r'path_to_your_people_folder
```

## 💡 How It Works

1. The script reads all images in your dataset folder.
2. Each face is encoded into a numeric vector using `face_recognition`.
3. Labels (person names) are encoded and converted to one-hot vectors.
4. A neural network is trained to classify faces based on these embeddings.
5. The webcam feed is captured with OpenCV:
   - Faces in each frame are detected and encoded.
   - Encoded faces are fed to the trained model for prediction.
   - A rectangle and predicted name are drawn on each detected face in real-time.

   5.3. A rectangle and predicted name are drawn on each detected face in real-time.



## 🚀 How to Run

Install dependencies first:
```bash
pip install -r requirements.txt
```

##### Run the script:
```bash
python your_script_name.py
```

