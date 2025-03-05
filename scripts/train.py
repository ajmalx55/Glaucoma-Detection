import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# ✅ Ensure TensorFlow sees GPU (if available)
print("🔍 Num GPUs Available:", len(tf.config.list_physical_devices('GPU')))

# ✅ Fix Dataset Path
BASE_DIR = "C:/Users/DELL/OneDrive/Desktop/E"  # Get root dir
TRAIN_DIR = os.path.join(BASE_DIR, "dataset/train")
VAL_DIR = os.path.join(BASE_DIR, "dataset/val")

# ✅ Check if dataset exists
if not os.path.exists(TRAIN_DIR) or not os.path.exists(VAL_DIR):
    raise FileNotFoundError("❌ Dataset not found! Place data inside 'dataset/train' and 'dataset/val'.")

# ✅ Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(128, 128),  # 👈 Matches model input shape!
    batch_size=32,
    class_mode="binary"
)

val_generator = val_datagen.flow_from_directory(
    VAL_DIR,
    target_size=(128, 128),
    batch_size=32,
    class_mode="binary"
)

# ✅ Define Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),  # 👈 Updated to match target_size
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation="sigmoid"),
])

# ✅ Compile Model
model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# ✅ Train Model
model.fit(
    train_generator,
    epochs=25,
    validation_data=val_generator
)

# ✅ Ensure models directory exists
models_dir = os.path.join(BASE_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

# ✅ Save the Model
model.save(os.path.join(models_dir, "glaucoma_detection_model.keras"))

print("✅ Training Completed! Model saved to 'models/glaucoma_detection_model.keras'")
