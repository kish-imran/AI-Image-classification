# ============================================================
# AI IMAGE CLASSIFICATION SYSTEM
# IMAGE PREDICTION
# ============================================================

import os
import sys
import numpy as np
import tensorflow as tf

from tensorflow.keras.preprocessing import image


# ============================================================
# 1. PROJECT PATH
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(
    BASE_DIR,
    "model",
    "image_classifier.keras"
)


# ============================================================
# 2. CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):
    print("\n" + "=" * 60)
    print("ERROR: TRAINED MODEL NOT FOUND")
    print("=" * 60)

    print("\nExpected model location:")
    print(MODEL_PATH)

    print("\nFirst run:")
    print("python train_model.py")

    sys.exit()


# ============================================================
# 3. LOAD MODEL
# ============================================================

print("\nLoading AI model...")

model = tf.keras.models.load_model(
    MODEL_PATH
)

print("Model loaded successfully!")


# ============================================================
# 4. IMAGE PATH
# ============================================================

if len(sys.argv) > 1:

    image_path = sys.argv[1]

else:

    image_path = input(
        "\nEnter the image path: "
    ).strip()


# ============================================================
# 5. CHECK IMAGE
# ============================================================

if not os.path.exists(image_path):

    print("\nERROR: Image not found!")

    print("You entered:")
    print(image_path)

    sys.exit()


# ============================================================
# 6. LOAD IMAGE
# ============================================================

print("\nProcessing image...")

img = image.load_img(
    image_path,
    target_size=(128, 128)
)


# ============================================================
# 7. CONVERT IMAGE TO ARRAY
# ============================================================

img_array = image.img_to_array(
    img
)


# ============================================================
# 8. NORMALIZE IMAGE
# ============================================================

img_array = img_array / 255.0


# ============================================================
# 9. ADD BATCH DIMENSION
# ============================================================

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ============================================================
# 10. MAKE PREDICTION
# ============================================================

prediction = model.predict(
    img_array,
    verbose=0
)


probability = float(
    prediction[0][0]
)


# ============================================================
# 11. DETERMINE CLASS
# ============================================================

if probability >= 0.5:

    predicted_class = "DOG"

    confidence = probability * 100

else:

    predicted_class = "CAT"

    confidence = (1 - probability) * 100


# ============================================================
# 12. DISPLAY RESULT
# ============================================================

print("\n" + "=" * 60)
print("IMAGE CLASSIFICATION RESULT")
print("=" * 60)

print("\nImage:")
print(image_path)

print("\nPrediction:")
print(predicted_class)

print("\nConfidence:")
print(f"{confidence:.2f}%")

print("\n" + "=" * 60)