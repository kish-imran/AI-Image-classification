# ============================================================
# AI IMAGE CLASSIFICATION SYSTEM
# CNN MODEL TRAINING
# ============================================================

import os

# Disable oneDNN optimization messages
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# ============================================================
# 1. PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "seg_train"
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "dataset",
    "seg_test"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "model"
)

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "image_classifier.keras"
)


# ============================================================
# 2. CREATE MODEL FOLDER
# ============================================================

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)


# ============================================================
# 3. PROJECT INFORMATION
# ============================================================

print("=" * 60)
print("AI IMAGE CLASSIFICATION SYSTEM")
print("=" * 60)

print()
print("TensorFlow Version:", tf.__version__)

print()
print("Checking dataset...")


# ============================================================
# 4. CHECK TRAINING FOLDER
# ============================================================

if not os.path.isdir(TRAIN_DIR):

    print()
    print("ERROR: Training folder not found!")
    print()
    print("Expected location:")
    print(TRAIN_DIR)

    exit()

print()
print("Training folder found!")
print(TRAIN_DIR)


# ============================================================
# 5. CHECK TESTING FOLDER
# ============================================================

if not os.path.isdir(TEST_DIR):

    print()
    print("ERROR: Testing folder not found!")
    print()
    print("Expected location:")
    print(TEST_DIR)

    exit()

print()
print("Testing folder found!")
print(TEST_DIR)


# ============================================================
# 6. IMAGE SETTINGS
# ============================================================

IMAGE_SIZE = (150, 150)

BATCH_SIZE = 32

EPOCHS = 10


# ============================================================
# 7. TRAINING DATA AUGMENTATION
# ============================================================

train_datagen = ImageDataGenerator(

    rescale=1.0 / 255,

    rotation_range=20,

    width_shift_range=0.2,

    height_shift_range=0.2,

    zoom_range=0.2,

    horizontal_flip=True

)


# ============================================================
# 8. TEST DATA PREPROCESSING
# ============================================================

test_datagen = ImageDataGenerator(

    rescale=1.0 / 255

)


# ============================================================
# 9. LOAD TRAINING DATA
# ============================================================

print()
print("=" * 60)
print("LOADING TRAINING DATA")
print("=" * 60)

train_data = train_datagen.flow_from_directory(

    TRAIN_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=True

)


# ============================================================
# 10. LOAD TESTING DATA
# ============================================================

print()
print("=" * 60)
print("LOADING TESTING DATA")
print("=" * 60)

test_data = test_datagen.flow_from_directory(

    TEST_DIR,

    target_size=IMAGE_SIZE,

    batch_size=BATCH_SIZE,

    class_mode="categorical",

    shuffle=False

)


# ============================================================
# 11. DATASET INFORMATION
# ============================================================

print()
print("=" * 60)
print("DATASET INFORMATION")
print("=" * 60)

print()
print("Training Images:", train_data.samples)

print("Testing Images:", test_data.samples)


# ============================================================
# 12. DISPLAY CLASSES
# ============================================================

print()
print("Classes detected:")

for class_name, class_number in train_data.class_indices.items():

    print(
        class_number,
        "->",
        class_name
    )


# ============================================================
# 13. NUMBER OF CLASSES
# ============================================================

NUM_CLASSES = len(
    train_data.class_indices
)

print()
print("Total Classes:", NUM_CLASSES)


# ============================================================
# 14. CHECK NUMBER OF CLASSES
# ============================================================

if NUM_CLASSES < 2:

    print()
    print("ERROR: At least 2 classes are required.")

    exit()


# ============================================================
# 15. CHECK CLASS MATCHING
# ============================================================

if train_data.class_indices != test_data.class_indices:

    print()
    print("=" * 60)
    print("ERROR: TRAINING AND TESTING CLASSES DO NOT MATCH")
    print("=" * 60)

    print()
    print("Training classes:")
    print(train_data.class_indices)

    print()
    print("Testing classes:")
    print(test_data.class_indices)

    exit()


# ============================================================
# 16. CREATE CNN MODEL
# ============================================================

print()
print("=" * 60)
print("CREATING CNN MODEL")
print("=" * 60)


model = models.Sequential([

    # Input
    layers.Input(
        shape=(150, 150, 3)
    ),

    # Convolution Layer 1
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Layer 2
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Layer 3
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Convolution Layer 4
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu"
    ),

    layers.MaxPooling2D(
        (2, 2)
    ),

    # Flatten
    layers.Flatten(),

    # Dense Layer
    layers.Dense(
        128,
        activation="relu"
    ),

    # Dropout
    layers.Dropout(
        0.5
    ),

    # Output
    layers.Dense(
        NUM_CLASSES,
        activation="softmax"
    )

])


# ============================================================
# 17. MODEL SUMMARY
# ============================================================

print()
print("=" * 60)
print("CNN MODEL SUMMARY")
print("=" * 60)

model.summary()


# ============================================================
# 18. COMPILE MODEL
# ============================================================

print()
print("=" * 60)
print("COMPILING MODEL")
print("=" * 60)

model.compile(

    optimizer="adam",

    loss="categorical_crossentropy",

    metrics=["accuracy"]

)

print()
print("Model compiled successfully!")


# ============================================================
# 19. TRAIN MODEL
# ============================================================

print()
print("=" * 60)
print("STARTING MODEL TRAINING")
print("=" * 60)

print()
print("Epochs:", EPOCHS)

print("Batch Size:", BATCH_SIZE)

print()
print("Training started...")


history = model.fit(

    train_data,

    epochs=EPOCHS,

    validation_data=test_data

)


# ============================================================
# 20. EVALUATE MODEL
# ============================================================

print()
print("=" * 60)
print("EVALUATING MODEL")
print("=" * 60)


test_loss, test_accuracy = model.evaluate(

    test_data,

    verbose=1

)


# ============================================================
# 21. DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("MODEL PERFORMANCE")
print("=" * 60)

print()

print(
    "Test Accuracy:",
    f"{test_accuracy * 100:.2f}%"
)

print(
    "Test Loss:",
    f"{test_loss:.4f}"
)


# ============================================================
# 22. TRAINING ACCURACY
# ============================================================

if "accuracy" in history.history:

    training_accuracy = history.history["accuracy"][-1]

    print(
        "Training Accuracy:",
        f"{training_accuracy * 100:.2f}%"
    )


# ============================================================
# 23. VALIDATION ACCURACY
# ============================================================

if "val_accuracy" in history.history:

    validation_accuracy = history.history["val_accuracy"][-1]

    print(
        "Validation Accuracy:",
        f"{validation_accuracy * 100:.2f}%"
    )


# ============================================================
# 24. SAVE MODEL
# ============================================================

print()
print("=" * 60)
print("SAVING MODEL")
print("=" * 60)


model.save(
    MODEL_PATH
)


# ============================================================
# 25. VERIFY MODEL
# ============================================================

if os.path.isfile(MODEL_PATH):

    print()
    print("MODEL SAVED SUCCESSFULLY!")

    print()
    print("Model location:")

    print(MODEL_PATH)

else:

    print()
    print("ERROR: Model was not saved.")


# ============================================================
# 26. DISPLAY CLASSES
# ============================================================

print()
print("=" * 60)
print("IMAGE CLASSIFICATION CLASSES")
print("=" * 60)

print()

for class_name, class_number in train_data.class_indices.items():

    print(
        class_number,
        "->",
        class_name
    )


# ============================================================
# 27. COMPLETED
# ============================================================

print()
print("=" * 60)
print("TRAINING PROCESS COMPLETED")
print("=" * 60)

print()
print("Your CNN image classification model is ready!")

print()
print("Model file:")

print(MODEL_PATH)

print()