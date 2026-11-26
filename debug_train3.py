"""
Minimal sanity test: can TensorFlow learn a simple 3-class problem
using the SAME images we're trying to train on?

If this works → the issue is with EfficientNetB0 specifically.
If this fails → the issue is with the data or labels.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
import numpy as np
import tensorflow as tf
import keras
from keras import layers, Model
from PIL import Image

ROOT = os.path.dirname(os.path.abspath(__file__))
pv   = config.PLANTVILLAGE_DIR
H, W = 128, 128  # smaller for speed

# Use 3 classes only for speed
all_classes = sorted([d for d in os.listdir(pv) if os.path.isdir(os.path.join(pv, d))])
classes = all_classes[:3]
print(f"Testing with 3 classes: {classes}")

X_train, y_train, X_val, y_val = [], [], [], []
for idx, cls in enumerate(classes):
    cls_dir = os.path.join(pv, cls)
    imgs = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg','.jpeg','.png'))][:100]
    for i, fname in enumerate(imgs):
        try:
            img = Image.open(os.path.join(cls_dir, fname)).convert("RGB")
            img = img.resize((H, W), Image.LANCZOS)
            arr = np.array(img).astype(np.float32) / 255.0  # simple [0,1] normalization
            if i < 80:
                X_train.append(arr); y_train.append(idx)
            else:
                X_val.append(arr); y_val.append(idx)
        except: pass

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
X_val   = np.array(X_val,   dtype=np.float32)
y_val   = np.array(y_val,   dtype=np.int32)
print(f"Train: {X_train.shape}  Val: {X_val.shape}")
print(f"Pixel range: [{X_train.min():.2f}, {X_train.max():.2f}]")
print(f"Labels: {np.unique(y_train)}")

# ── Simple CNN (NOT EfficientNet) ────────────────────────────────────────────
model = keras.Sequential([
    keras.Input(shape=(H, W, 3)),
    layers.Conv2D(32, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation="relu", padding="same"),
    layers.MaxPooling2D(),
    layers.Conv2D(128, 3, activation="relu", padding="same"),
    layers.GlobalAveragePooling2D(),
    layers.Dense(64, activation="relu"),
    layers.Dropout(0.3),
    layers.Dense(3, activation="softmax"),
])
model.compile(optimizer=keras.optimizers.Adam(1e-3),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("\nTraining simple CNN for 10 epochs...")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=10, batch_size=32, verbose=1)

final_train = history.history["accuracy"][-1]
final_val   = history.history["val_accuracy"][-1]
print(f"\nSimple CNN — Train: {final_train*100:.1f}%  Val: {final_val*100:.1f}%")

with open(os.path.join(ROOT, "debug_train3_result.log"), "w") as f:
    f.write(f"Simple CNN (not EfficientNet)\n")
    f.write(f"Train acc: {final_train*100:.1f}%\n")
    f.write(f"Val acc  : {final_val*100:.1f}%\n")
    f.write(f"Verdict: {'DATA IS LEARNABLE' if final_val > 0.50 else 'DATA PROBLEM'}\n")
print("Result saved to debug_train3_result.log")
