import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

import os

img_path = "C:/Users/DELL/OneDrive/Desktop/Glaucoma-Detection/test.png"

# Check if the file exists
if os.path.exists(img_path):
    print("✅ File found!")
else:
    print("❌ File NOT found!")


# Load and preprocess image
img = image.load_img(img_path, target_size=(128, 128))  # Resize to match model input
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array /= 255.0  # Normalize

# Load model
model_path = r"C:\Users\DELL\OneDrive\Desktop\Glaucoma-Detection\models\glaucoma_detection_model.keras"  # Update this path
model = tf.keras.models.load_model(model_path)

# Predict
prediction = model.predict(img_array)
print("Prediction:", prediction)
# Assuming the model outputs a probability (sigmoid activation)
threshold = 0.5  # Usually, 0.5 is used as the decision threshold

# Conditional statement for classification
if prediction[0][0] >= threshold:
    print("🛑 Glaucoma Detected! Please consult an eye specialist.")
else:
    print("✅ No Glaucoma Detected. Eye appears normal.")