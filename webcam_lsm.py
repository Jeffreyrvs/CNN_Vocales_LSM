""""
Inferencia en tiempo real — Vocales LSM
Captura webcam → detecta mano (MediaPipe Tasks API) → recorta 240x240 → predice vocal.

Dependencias:
    pip install opencv-python mediapipe tensorflow

Teclas:
    q  — salir
"""

import os
import urllib.request
import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

# ── Configuración 
MODEL_PATH  = os.path.join(os.path.dirname(__file__), "mejor_modelo_lsm.keras")
IMG_SIZE    = (224, 224)
CLASES      = ["A", "E", "I", "O", "U"]
CROP_SHOW   = 240
INFER_EVERY = 4

# Modelo de detección de manos de MediaPipe (se descarga una sola vez ~2 MB)
HAND_MODEL  = os.path.join(os.path.dirname(__file__), "hand_landmarker.task")
HAND_URL    = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)

# Conexiones del esqueleto de la mano (índices de landmarks) mediapipe
HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# descargar modelo de MediaPipe si no existe 
if not os.path.exists(HAND_MODEL):
    print("Descargando hand_landmarker.task (~2 MB)…")
    urllib.request.urlretrieve(HAND_URL, HAND_MODEL)
    print("Modelo descargado.\n")

# cargar modelo entrenado 
print("Cargando modelo LSM…")
model = tf.keras.models.load_model(MODEL_PATH)
print("Modelo listo.\n")

#inferencia 
def predecir_imagen(bgr_crop):
    gray    = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, IMG_SIZE)
    arr     = resized.astype("float32") / 255.0
    arr     = np.expand_dims(arr, axis=(0, -1))
    probs   = model.predict(arr, verbose=0)[0]
    idx     = int(np.argmax(probs))
    return CLASES[idx], float(probs[idx])

# ── Inicializar MediaPipe HandLandmarker ──────────────────────────────────────
options = mp_vision.HandLandmarkerOptions(
    base_options=mp_python.BaseOptions(model_asset_path=HAND_MODEL),
    num_hands=1,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.5,
    min_tracking_confidence=0.5,
)
landmarker = mp_vision.HandLandmarker.create_from_options(options)

# funciones para recortar y rellenar imagenes
def bbox_ajustado(landmarks, frame_w, frame_h, padding=0.10):
    """Bounding box ajustado al perímetro de la mano con padding mínimo."""
    xs = [lm.x for lm in landmarks]
    ys = [lm.y for lm in landmarks]
    x1, x2 = int(min(xs) * frame_w), int(max(xs) * frame_w)
    y1, y2 = int(min(ys) * frame_h), int(max(ys) * frame_h)

    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(frame_w, x2 + pad_x)
    y2 = min(frame_h, y2 + pad_y)
    return x1, y1, x2, y2

def letterbox(bgr_crop, size=224):
    """
    Escala el recorte para que quepa en size×size manteniendo la relación de
    aspecto, luego centra sobre un canvas negro de size×size.
    """
    h, w = bgr_crop.shape[:2]
    scale  = size / max(h, w)
    new_w  = int(w * scale)
    new_h  = int(h * scale)
    resized = cv2.resize(bgr_crop, (new_w, new_h), interpolation=cv2.INTER_AREA)

    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    y_off  = (size - new_h) // 2
    x_off  = (size - new_w) // 2
    canvas[y_off:y_off + new_h, x_off:x_off + new_w] = resized
    return canvas

def dibujar_esqueleto(frame, landmarks, frame_w, frame_h):
    pts = [(int(lm.x * frame_w), int(lm.y * frame_h)) for lm in landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 200, 255), 1)
    for p in pts:
        cv2.circle(frame, p, 3, (255, 255, 255), -1)

# ── Captura principal de frames
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("No se pudo abrir la cámara.")

clase_actual = "—"
conf_actual  = 0.0
frame_idx    = 0

print("Presiona  q  para salir.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame    = cv2.flip(frame, 1)
    h, w     = frame.shape[:2]
    rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result   = landmarker.detect(mp_image)

    crop_vista = None

    if result.hand_landmarks:
        lms = result.hand_landmarks[0]
        x1, y1, x2, y2 = bbox_ajustado(lms, w, h)

        crop_raw    = frame[y1:y2, x1:x2]
        crop_modelo = letterbox(crop_raw, size=IMG_SIZE[0])   # 224×224 con padding negro
        crop_vista  = cv2.resize(crop_modelo, (CROP_SHOW, CROP_SHOW))

        if frame_idx % INFER_EVERY == 0 and crop_raw.size > 0:
            clase_actual, conf_actual = predecir_imagen(crop_modelo)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 220, 0), 2)
        dibujar_esqueleto(frame, lms, w, h)

    # texto en la esquina
    etiqueta = f"{clase_actual}  {conf_actual*100:.0f}%"
    cv2.rectangle(frame, (0, 0), (230, 55), (0, 0, 0), -1)
    cv2.putText(frame, etiqueta, (10, 42),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 220, 0), 2, cv2.LINE_AA)

    cv2.imshow("LSM Vocales — q para salir", frame)

    if crop_vista is not None:
        cv2.imshow(f"Recorte {CROP_SHOW}x{CROP_SHOW}", crop_vista)

    frame_idx += 1
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
landmarker.close()
