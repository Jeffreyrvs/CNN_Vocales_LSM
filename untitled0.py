


"""
Reconocimiento de Vocales en Lengua de Señas Mexicana (LSM)
CNN para clasificar: A, E, I, O, U

Estructura esperada del dataset:
    dataset/
    ├── train/
    │   ├── A/  ├── E/  ├── I/  ├── O/  └── U/
    ├── test/
    │   ├── A/  ├── E/  ├── I/  ├── O/  └── U/
    └── validation/
        ├── A/  ├── E/  ├── I/  ├── O/  └── U/
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns


# CONFIGURACIÓN

IMG_SIZE    = (224, 224)
CHANNELS    = 1         
BATCH_SIZE  = 32
EPOCHS      = 30
LR          = 0.001
NUM_CLASES  = 5
CLASES      = ['A', 'E', 'I', 'O', 'U']


DATASET_DIR = "./dataset"

TRAIN_DIR = os.path.join(DATASET_DIR, "train_224")
VAL_DIR   = os.path.join(DATASET_DIR, "val_224")
TEST_DIR  = os.path.join(DATASET_DIR, "test_224")

COLOR_MODE = "rgb" if CHANNELS == 3 else "grayscale"


# GENERADORES DE DATOS CON AUGMENTATION

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True,
)

val_test_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=CLASES,
    shuffle=True,
    seed=42,
)

val_gen = val_test_datagen.flow_from_directory(
    VAL_DIR,
    target_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=CLASES,
    shuffle=False,
)

test_gen = val_test_datagen.flow_from_directory(
    TEST_DIR,
    target_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",
    classes=CLASES,
    shuffle=False,
)

print("\n Clases detectadas:", train_gen.class_indices)
print(f"   Entrenamiento : {train_gen.samples} imágenes")
print(f"   Validación    : {val_gen.samples} imágenes")
print(f"   Prueba        : {test_gen.samples} imágenes\n")


# ARQUITECTURA CNN

def build_model(input_shape, num_classes):
    model = models.Sequential([
        # Bloque 1
        layers.Conv2D(32, (3, 3), activation="relu", padding="same",
                      input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),

        # Bloque 2
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Bloque 3
        layers.Conv2D(128, (3, 3), activation="relu", padding="same"),
        layers.MaxPooling2D((2, 2)),

        # Clasificador
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model

input_shape = (*IMG_SIZE, CHANNELS)
model = build_model(input_shape, NUM_CLASES)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LR),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()


# CALLBACKS

callbacks = [
    ModelCheckpoint(
        "mejor_modelo_lsm.keras",
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
    ),
    EarlyStopping(
        monitor="val_accuracy",
        patience=7,
        restore_best_weights=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    ),
]

# ─────────────────────────────────────────────# ENTRENAMIENTO

history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=callbacks,
)

# graficas
def plot_history(history):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy
    axes[0].plot(history.history["accuracy"],     label="Train")
    axes[0].plot(history.history["val_accuracy"], label="Validación")
    axes[0].set_title("Precisión por época")
    axes[0].set_xlabel("Época")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True)

    # Loss
    axes[1].plot(history.history["loss"],     label="Train")
    axes[1].plot(history.history["val_loss"], label="Validación")
    axes[1].set_title("Pérdida por época")
    axes[1].set_xlabel("Época")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig("curvas_entrenamiento.png", dpi=150)
    plt.show()
    print("Curvas guardadas en curvas_entrenamiento.png")

plot_history(history)

#tabla 
print("\n─── Evaluación en conjunto de prueba ───")
test_loss, test_acc = model.evaluate(test_gen)
print(f"Loss : {test_loss:.4f}")
print(f"Acc  : {test_acc:.4f}")

# Predicciones
test_gen.reset()
y_pred_probs = model.predict(test_gen)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_gen.classes

# Reporte detallado
print("\n─── Reporte de clasificación ───")
print(classification_report(y_true, y_pred, target_names=CLASES))

# Matriz de confusión
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=CLASES, yticklabels=CLASES)
plt.title("Matriz de Confusión — Conjunto de Prueba")
plt.ylabel("Etiqueta real")
plt.xlabel("Predicción")
plt.tight_layout()
plt.savefig("matriz_confusion.png", dpi=150)
plt.show()
print("Matriz guardada en matriz_confusion.png")

# MODELO FINAL

model.save("modelo_lsm_vocales_final.keras")
print("\n Modelo guardado en modelo_lsm_vocales_final.keras")
