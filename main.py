# ============================================================
# AI IMAGE CLASSIFICATION SYSTEM
# MAIN PROGRAM
# ============================================================

import os
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
# 2. DISPLAY HEADER
# ============================================================

def show_header():

    print("\n" + "=" * 60)
    print("          AI IMAGE CLASSIFICATION SYSTEM")
    print("=" * 60)


# ============================================================
# 3. LOAD MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):

        print("\nERROR: Trained model not found!")

        print("\nPlease train the model first:")
        print("python train_model.py")

        return None

    print("\nLoading AI model...")

    model = tf.keras.models.load_model(
        MODEL_PATH
    )

    print("Model loaded successfully!")

    return model


# ============================================================
# 4. PREDICT IMAGE
# ============================================================

def predict_image(model):

    print("\n" + "-" * 60)
    print("IMAGE PREDICTION")
    print("-" * 60)

    image_path = input(
        "\nEnter image path: "
    ).strip()

    # Check image
    if not os.path.exists(image_path):

        print("\nERROR: Image not found!")

        return

    try:

        # Load image
        img = image.load_img(
            image_path,
            target_size=(128, 128)
        )

        # Convert image to array
        img_array = image.img_to_array(
            img
        )

        # Normalize
        img_array = img_array / 255.0

        # Add batch dimension
        img_array = np.expand_dims(
            img_array,
            axis=0
        )

        # Make prediction
        prediction = model.predict(
            img_array,
            verbose=0
        )

        probability = float(
            prediction[0][0]
        )

        # Determine class
        if probability >= 0.5:

            predicted_class = "DOG"

            confidence = probability * 100

        else:

            predicted_class = "CAT"

            confidence = (1 - probability) * 100

        # Display result
        print("\n" + "=" * 60)
        print("             CLASSIFICATION RESULT")
        print("=" * 60)

        print("\nImage:")
        print(image_path)

        print("\nPredicted Class:")
        print(predicted_class)

        print("\nConfidence:")
        print(f"{confidence:.2f}%")

        print("\n" + "=" * 60)

    except Exception as error:

        print("\nERROR while processing image:")
        print(error)


# ============================================================
# 5. MAIN MENU
# ============================================================

def main():

    show_header()

    model = load_model()

    if model is None:

        return

    while True:

        print("\n")
        print("=" * 60)
        print("MAIN MENU")
        print("=" * 60)

        print("\n1. Classify an Image")
        print("2. Exit")

        choice = input(
            "\nEnter your choice: "
        ).strip()

        if choice == "1":

            predict_image(model)

        elif choice == "2":

            print("\nThank you for using")
            print("AI Image Classification System!")

            break

        else:

            print("\nInvalid choice!")
            print("Please enter 1 or 2.")


# ============================================================
# 6. PROGRAM START
# ============================================================

if __name__ == "__main__":

    main()