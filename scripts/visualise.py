import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix
import os

# Load model
model = tf.keras.models.load_model("../models/glaucoma_detection_model.keras")

# Load validation dataset
val_dir = "dataset/val"  # Adjust if needed
img_size = (128, 128)  # Ensure it matches your training setup

val_generator = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir, image_size=img_size, batch_size=32
)

# Get predictions
y_true = np.concatenate([y for x, y in val_generator], axis=0)
y_pred = model.predict(val_generator)
y_pred = (y_pred > 0.5).astype(int)

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Normal", "Glaucoma"], yticklabels=["Normal", "Glaucoma"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
