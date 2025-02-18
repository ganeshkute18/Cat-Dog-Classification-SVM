import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("cats_vs_dogs.h5")

# Define function to preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))  # Resize to match model input size
    image = image / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Expand dimensions to fit model input
    return image

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Predict on the current frame
    img = preprocess_image(frame)
    prediction = model.predict(img)[0][0]

    # Convert prediction to label
    label = "Dog" if prediction > 0.5 else "Cat"

    # Display prediction on frame
    cv2.putText(frame, f"Prediction: {label}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow("Cat vs Dog Classifier", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
