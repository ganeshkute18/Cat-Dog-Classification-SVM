import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load trained model
model = load_model("cats_vs_dogs.h5")
class_labels = ["Cat", "Dog"]

# Open Webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess Image
    img = cv2.resize(frame, (224, 224))
    img = img / 255.0  # Normalize
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict class
    prediction = model.predict(img)[0][0]  # Get prediction value
    predicted_class = class_labels[int(prediction > 0.5)]

    # Display Prediction
    cv2.putText(frame, f"Prediction: {predicted_class}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Cat vs Dog Classifier", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
