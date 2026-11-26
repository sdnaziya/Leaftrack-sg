"""
Debug training approach 2: Don't freeze the base at all.
Train the entire EfficientNetB0 + head from the start with a low LR.
This avoids the frozen-BatchNorm problem entirely.
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
H, W = config.IMAGE_SIZE

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
            arr = (arr / 127.5) - 1.0
            if i < 50:
                X_train.append(arr); y_train.append(idx)
            else:
                X_val.append(arr); y_val.append(idx)
        except: pass

X_train = np.array(X_train, dtype=np.float32)
y_train = np.array(y_train, dtype=np.int32)
X_val   = np.array(X_val,   dtype=np.float32)
y_val   = np.array(y_val,   dtype=np.int32)
print(f"Train: {X_train.shape}  Val: {X_val.shape}")

# ── Approach: full fine-tuning from the start ────────────────────────────────
base_model = EfficientNetB0(weights="imagenet", include_top=False, input_shape=(H,W,3))
# Do NOT freeze — let all weights update
base_model.trainable = True

inputs = keras.Input(shape=(H,W,3))
x = base_model(inputs)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(len(classes), activation="softmax")(x)
model = Model(inputs, outputs)

# Use a VERY low LR for full fine-tuning
model.compile(optimizer=keras.optimizers.Adam(1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

print("\nTraining 8 epochs (full fine-tuning, lr=1e-5)...")
history = model.fit(X_train, y_train,
                    validation_data=(X_val, y_val),
                    epochs=8, batch_size=16, verbose=1)

final_train_acc = history.history["accuracy"][-1]
final_val_acc   = history.history["val_accuracy"][-1]
print(f"\nFinal train accuracy: {final_train_acc*100:.1f}%")
print(f"Final val accuracy  : {final_val_acc*100:.1f}%")

with open(os.path.join(ROOT, "debug_train2_result.log"), "w") as f:
    f.write(f"Approach: Full fine-tuning, lr=1e-5\n")
    f.write(f"Train acc: {final_train_acc*100:.1f}%\n")
    f.write(f"Val acc  : {final_val_acc*100:.1f}%\n")
    f.write(f"Verdict: {'LEARNING' if final_val_acc > 0.30 else 'NOT LEARNING'}\n")
print("Result saved to debug_train2_result.log")
