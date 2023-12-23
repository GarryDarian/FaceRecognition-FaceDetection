import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from collections import Counter
from sklearn.preprocessing import LabelEncoder

# Function untuk apply CLAHE ke image

def apply_clahe(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    return enhanced_image

# Function untuk detect faces menggunakan HaarCascade
def detect_faces(image, face_cascade):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
    return faces

# Function untuk train and test the model
def train_and_test_model(train_images, train_labels, test_images, test_labels, save_path=None):
    face_recognizer = cv2.face.LBPHFaceRecognizer_create()

    faces = []
    labels = []

    label_encoder = LabelEncoder()
    integer_labels = label_encoder.fit_transform(train_labels)

    for i, image in enumerate(train_images):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces.append(gray_image)
        labels.append(integer_labels[i])

    face_recognizer.train(faces, np.array(labels, dtype=np.int32))

    if save_path:
        face_recognizer.save(save_path)

    predicted_labels = []
    actual_labels = []
    confidences = []

    for i, image in enumerate(test_images):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        label, confidence = face_recognizer.predict(gray_image)
        predicted_labels.append(label)
        actual_labels.append(integer_labels[i])
        confidences.append(confidence)

    accuracy = accuracy_score(actual_labels, predicted_labels)
    overall_accuracy = (1 - (np.mean(confidences) / 300)) * 100

    print("Training model...")
    print("Done Training!")
    print(f"Overall Accuracy: {overall_accuracy:.2f}%")

    return face_recognizer, label_encoder, overall_accuracy

dataset_path = 'Dataset'

image_paths = []
labels = []

for athlete_folder in os.listdir(dataset_path):
    athlete_path = os.path.join(dataset_path, athlete_folder)
    
    for image_file in os.listdir(athlete_path):
        image_path = os.path.join(athlete_path, image_file)
        image_paths.append(image_path)
        labels.append(athlete_folder)

# Split dataset jadi training and testing sets ketika maintaining class distribution
train_image_paths, test_image_paths, train_labels, test_labels = train_test_split(
    image_paths, labels, test_size=0.2, random_state=42, stratify=labels
)

train_images = [apply_clahe(cv2.imread(path)) for path in train_image_paths] 
test_images = [apply_clahe(cv2.imread(path)) for path in test_image_paths]

# print(f"Number of training samples: {len(train_images)}")
# print(f"Number of testing samples: {len(test_images)}")

train_label_counts = Counter(train_labels)
# print("\nTraining set class distribution:")
# print(train_label_counts)

test_label_counts = Counter(test_labels)
# print("\nTesting set class distribution:")
# print(test_label_counts)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

face_recognizer, label_encoder, overall_accuracy = train_and_test_model(
    train_images, train_labels, test_images, test_labels
)

# Function untuk predict dengan saved model
def predict_with_model(model, label_encoder, input_image_path):
    face_recognizer = model

    try:
        input_image = cv2.imread(input_image_path)
        if input_image is None:
            print(f"Error: Unable to read the image at '{input_image_path}'.")
            return
    except Exception as e:
        print(f"Error: {e}")
        return

    # ubah input image to grayscale
    gray_input_image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)

    # Detect faces using HaarCascade
    faces = detect_faces(input_image, face_cascade)

    if faces is not None:
        for (x, y, w, h) in faces:
            face_roi = gray_input_image[y:y + h, x:x + w]
            label, confidence = face_recognizer.predict(face_roi)

            confidence_percentage = (1 - (confidence / 300)) * 100

            predicted_name = label_encoder.inverse_transform([label])[0]

            cv2.putText(input_image, f"{predicted_name} ({confidence_percentage:.2f}%)", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            cv2.rectangle(input_image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # height, width, _ = input_image.shape
        # new_width = int(width * 0.8)
        # new_height = int(height * 0.8)
        # resized_image = cv2.resize(input_image, (new_width, new_height))

        # # Create a named window and set its properties
        # cv2.namedWindow("Prediction Result", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Prediction Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        # cv2.namedWindow("Prediction Result", cv2.WINDOW_NORMAL)
        # cv2.setWindowProperty("Prediction Result", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        cv2.imshow("Prediction Result", input_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("Error: No face detected in the input image.")

# Main program
while True:
    print("Basketball Player Face Detection")
    print("1. Train model")
    print("2. Predict")
    print("3. Exit")

    choice = input("Choose your option: ")

    if choice == "1":
        # Train and test the model
        model_save_path = "trained_model.yml"  # Change this path to your desired location
        face_recognizer, label_encoder, overall_accuracy = train_and_test_model(
            train_images, train_labels, test_images, test_labels, save_path=model_save_path
        )
        input("Press enter to continue...")
    elif choice == "2":
        # Predict using the saved model
        model_path = "trained_model.yml"  # Change this path to the saved model
        if os.path.exists(model_path):
            input_image_path = input("Input image path (absolute path) >> ")
            predict_with_model(face_recognizer, label_encoder, input_image_path)
        else:
            print("Error: Trained model not found. Please train the model first.")
    elif choice == "3":
        break
    else:
        print("Invalid option. Please choose a valid option.")
