
import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.regularizers import l2

dataset_path = r"C:\Users\user\Desktop\BirdVsDroneVsAirplane"
classes = ["Aeroplanes", "Birds", "Drones"]
img_size = 128

data, labels = [], []
skipped = 0
print("Загрузка изображений...")
for class_idx, class_name in enumerate(classes):
    class_dir = os.path.join(dataset_path, class_name)
    for fname in os.listdir(class_dir):
        fpath = os.path.join(class_dir, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with Image.open(fpath) as img:
                img = img.convert('L').resize((img_size, img_size))
                data.append(np.array(img))
                labels.append(class_idx)
        except:
            skipped += 1

print(f"Загружено: {len(data)} | Пропущено: {skipped}")
print(f"Распределение по классам: {np.bincount(labels)}")

X = np.array(data).reshape(-1, img_size, img_size, 1) / 255.0
Y = to_categorical(labels, num_classes=len(classes))

X_temp, X_test, Y_temp, Y_test = train_test_split(
    X, Y, test_size=0.1, random_state=2, stratify=labels)
X_train, X_val, Y_train, Y_val = train_test_split(
    X_temp, Y_temp, test_size=0.15, random_state=2, stratify=np.argmax(Y_temp, axis=1))

print(f"Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")

plt.figure(figsize=(10, 4))
for i in range(5):
    plt.subplot(1, 5, i+1)
    plt.imshow(X_train[i].reshape(img_size, img_size), cmap='gray')
    plt.title(classes[np.argmax(Y_train[i])])
    plt.axis('off')
plt.suptitle("Примеры из обучающей выборки")
plt.show()

y_train_labels = np.argmax(Y_train, axis=1)
class_weights = compute_class_weight(
    class_weight='balanced',
    classes=np.unique(y_train_labels),
    y=y_train_labels
)
class_weight_dict = dict(enumerate(class_weights))
print("Веса классов:", class_weight_dict)

model = Sequential([
    Conv2D(32, (3,3), padding='same', kernel_regularizer=l2(0.001),
           input_shape=(img_size, img_size, 1)),
    tf.keras.layers.Activation('relu'),
    MaxPool2D((2,2)),
    Dropout(0.25),

    Conv2D(64, (3,3), padding='same', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Activation('relu'),
    MaxPool2D((2,2)),
    Dropout(0.25),

    Conv2D(128, (3,3), padding='same', kernel_regularizer=l2(0.001)),
    tf.keras.layers.Activation('relu'),
    MaxPool2D((2,2)),
    Dropout(0.3),

    Flatten(),
    Dense(256, kernel_regularizer=l2(0.001)),
    tf.keras.layers.Activation('relu'),
    Dropout(0.5),
    Dense(len(classes), activation='softmax')
])

model.summary()

model.compile(
    optimizer=Adam(learning_rate=0.0005),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

datagen = ImageDataGenerator(
    rotation_range=15,
    zoom_range=0.15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    horizontal_flip=True,
    fill_mode='nearest'
)
datagen.fit(X_train)

lr_reduction = ReduceLROnPlateau(monitor='val_loss', patience=4, factor=0.5, min_lr=1e-6)
early_stop = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

history = model.fit(
    datagen.flow(X_train, Y_train, batch_size=32, shuffle=True),
    epochs=30,
    validation_data=(X_val, Y_val),
    callbacks=[lr_reduction, early_stop],
    class_weight=class_weight_dict
)

plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.legend(); plt.title('Loss')
plt.subplot(1,2,2)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.legend(); plt.title('Accuracy')
plt.show()

Y_pred = model.predict(X_test)
Y_pred_classes = np.argmax(Y_pred, axis=1)
Y_true = np.argmax(Y_test, axis=1)

print("\nClassification Report (Validation):")
print(classification_report(Y_true, Y_pred_classes, target_names=classes, zero_division=0))

cm = confusion_matrix(Y_true, Y_pred_classes)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', xticklabels=classes, yticklabels=classes)
plt.xlabel('Predicted'); plt.ylabel('True'); plt.title('Confusion Matrix')
plt.show()

print(model.summary())

def predict_new_image(image_path):
    try:
        with Image.open(image_path) as img:
            img = img.convert('L').resize((img_size, img_size))
            img_array = np.array(img)
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        return
    img_input = img_array.reshape(1, img_size, img_size, 1) / 255.0
    pred = model.predict(img_input, verbose=0)[0]
    pred_class = np.argmax(pred)
    conf = pred[pred_class] * 100

    plt.figure()
    plt.imshow(img_array, cmap='gray')
    plt.title(f"{classes[pred_class]} ({conf:.2f}%)")
    plt.axis('off')
    plt.show()

    print(f"Предсказан: {classes[pred_class]}")
    print(f"Вероятности: {dict(zip(classes, np.round(pred*100, 2)))}%")

test_images = [
    r"C:\Users\user\Desktop\test\17236_1645150699.jpg",
    r"C:\Users\user\Desktop\test\aerial-drone-with-camera-hovering-in-clear-blue-sky-during-daytime-cut-out-transparent-png.png",
    r"C:\Users\user\Desktop\test\beautiful-bird-sky_33755-6271.jpg",
    r"C:\Users\user\Desktop\test\bird-flying-sky_849832-132.jpg",
    r"C:\Users\user\Desktop\test\DJI-Phantom-4-Review-Cinematic-Aerial-Cinematography.00_00_16_19.Still001.jpg",
    r"C:\Users\user\Desktop\test\i.jfif"
]

for img_path in test_images:
    if os.path.exists(img_path):
        predict_new_image(img_path)
    else:
        print(f"Файл не найден: {img_path}")
