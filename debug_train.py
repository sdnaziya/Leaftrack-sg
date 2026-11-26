"""
Minimal training debug — train on 50 images per class for 5 epochs
and see if val_accuracy goes above 20%. If it stays at 10%, there's a
pipeline bug. If it rises, the full training just needs more time.

Run: venv\Scripts\python.exe debug_train.py
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import numpy as np
import tensorflow as tf
import keras
from keras.applications import EfficientNetB0
from keras import layers, Model
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
pv   = config.PLANTVILLAGE_DIR
H, W = config.IMAGE_SIZE   # (224, 224)

# ── Load 50 images per class into memory ─────────────────────────────────────
classes = sorted([d for d in os.listdir(pv) if os.path.isdir(os.path.join(pv, d))])
print(f"Classes: {len(classes)}")

X_train, y_train, X_val, y_val = [], [], [], []
for idx, cls in enumerate(classes):
    cls_dir = os.path.join(pv, cls)
    imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))][:60]
    for i, fname in enumerate(imgs):
        try:
            img = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
            img = img.resize((H, W), Image.LANCZOS)
            arr = np.array(img).astype(np.float32)
            # EfficientNet preprocessing
            arr = (arr / 127.5) - 1.0
            if i < 50:
                X_train.append(arr); y_train.append(idx)
            else:
                X_val.append(arr); y_val.append(idx)
        except:
            pass

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
X_val   = np.array(X_val,   dtype=np.float32)
y_val   = np.array(y_val,   dtype=np.int32)
print(f"Train: {X_train.shape}  Val: {X_val.shape}")
print(f"Train pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"Train label range: [{y_train.min()}, {y_train.max()}]")

# ── Build model ──────────────────────────────────────────────────────────────
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(H,W,3))
base_model.trainable = False

# CRITICAL FIX: Frozen BatchNorm in EfficientNetB0 causes the model to output
# near-constant features when the base is frozen with training=False.
# Solution: unfreeze ONLY the BatchNormalization layers inside EfficientNetB0
# so they can adapt their moving statistics to our dataset, while keeping
# all conv weights frozen.
for layer in base_model.layers:
    if isinstance(layer, (keras.layers.BatchNormalization,
                           tf.keras.layers.BatchNormalization)):
        layer.trainable = True

inputs = keras.Input(shape=(H,W,3))
x = base_model(inputs, training=True)  # training=True so BN updates stats
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(classes), activation="softmax")(x)
model = Model(inputs, outputs)
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# ── Train for 5 epochs ────────────────────────────────────────────────────────
print("\nTraining 5 epochs on 500 images...")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=5, batch_size=32, verbose=1)

# ── Report ────────────────────────────────────────────────────────────────────
final_train_acc = history.history["accuracy"][-1]
final_val_acc   = history.history["val_accuracy"][-1]
print(f"\nFinal train accuracy: {final_train_acc*100:.1f}%")
print(f"Final val accuracy  : {final_val_acc*100:.1f}%")

if final_val_acc > 0.30:
    print("\n✓ Model IS learning — the pipeline works.")
    print("  The full training just needs to run longer.")
    print("  Run: venv\\Scripts\\python.exe retrain.py")
else:
    print("\n✗ Model is NOT learning — pipeline bug detected.")
    print("  Check: preprocessing, label mapping, loss function.")

# Save result
with open(os.path.join(ROOT, "debug_train_result.log"), "w") as f:
    f.write(f"Train acc: {final_train_acc*100:.1f}%\n")
    f.write(f"Val acc  : {final_val_acc*100:.1f}%\n")
    f.write(f"Verdict: {'LEARNING' if final_val_acc > 0.30 else 'NOT LEARNING'}\n")
print("\nResult saved to debug_train_result.log")
