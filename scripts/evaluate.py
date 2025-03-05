import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Load the trained model
model_path = "../models/glaucoma_detection_model.keras"
model = tf.keras.models.load_model(model_path)
print("✅ Model loaded successfully!")
model.summary()  # Debugging step to verify input shape

# Define dataset paths
test_dir = "dataset/test"  # Ensure this path exists

# Image Preprocessing
test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(128, 128),  # 👈 Match it with the first Conv2D output
    batch_size=1,  # Debugging: set batch size to 1
    class_mode="binary",
    shuffle=False
)

# Evaluate the model
try:
    loss, accuracy = model.evaluate(test_generator)
    print(f"📊 Test Accuracy: {accuracy * 100:.2f}%")
    print(f"📉 Test Loss: {loss:.4f}")
except ValueError as e:
    print(f"❌ Error in model evaluation: {e}")
