import os
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import cv2
import face_recognition
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

people_dataset = r'D:\Work\Projects\AI\face_recognitio(CNN)\people'
names = os.listdir(people_dataset)
people_names = []
people_images_paths = []
people_images_array = []
people_dataset_encodes = []
valid_extensions = ['.jpg', '.jpeg', '.png']

for i in names:
    ext = os.path.splitext(i)[1].lower()
    if ext in valid_extensions:
        people_names.append(os.path.splitext(i)[0])
        people_images_paths.append(os.path.join(people_dataset, i))

for j in people_images_paths:
    imgs = cv2.imread(j)
    people_images_array.append(cv2.resize(imgs, (128, 128)))


def encoding_dataset():
    for img in people_images_array:
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        locations = face_recognition.face_locations(rgb_img)
        encodings = face_recognition.face_encodings(rgb_img, locations)
        if len(encodings) > 0:
            people_dataset_encodes.append(encodings[0])
        else:
            print("No face found in this image.")


encoding_dataset()

labels_decoder = LabelEncoder()
labels_encoded = labels_decoder.fit_transform(people_names)
labels_onehot_encoded = to_categorical(labels_encoded)  # أبسط وأحدث طريقة

people_dataset_encodes = np.array(people_dataset_encodes)  # مهم تحويلها لمصفوفة numpy

model = Sequential([
    Dense(64, activation='relu', input_shape=(128,)),
    Dense(32, activation='relu'),
    Dense(labels_onehot_encoded.shape[1], activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(people_dataset_encodes, labels_onehot_encoded, epochs=20, batch_size=12)

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    smalled_img = cv2.resize(frame, (0, 0), fx=0.50, fy=0.50)
    rgb_img = cv2.cvtColor(smalled_img, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(smalled_img, face_locations)

    if face_encodings:  # تأكد في وجوه
        # ✅ اعمل predict لكل الوجوه دفعة وحدة
        results = model.predict(np.array(face_encodings))  # شكلها [عدد_الوجوه, عدد_الأشخاص]

        for result, face_loc in zip(results, face_locations):
            top, right, bottom, left = face_loc
            top *= 2
            right *= 2
            bottom *= 2
            left *= 2

            final_result = np.argmax(result)  # index تبع أعلى احتمال
            predicted_name = people_names[final_result]

            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
            cv2.putText(frame, predicted_name, (left, top - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

