from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import load_model
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split

MAX_POINTS = 100
IMAGE_SIZE = 256
SIGMA = 2


def load_data(dataset_dir):
    images = []
    points = []

    for filename in os.listdir(dataset_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(dataset_dir, filename)
            annotation_path = os.path.join(
                dataset_dir, filename.replace('.png', '_points.txt'))

            img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)

            with open(annotation_path, 'r') as f:
                coords = [list(map(float, line.strip().split(',')))
                          for line in f.readlines()]
                points.append(coords)

    return np.array(images), points


X, y = load_data('./dataset')


def adjust_labels_scaled(annotations, original_size=500, new_size=IMAGE_SIZE):
    adjusted_annotations = []
    scale = new_size / original_size
    for annotation in annotations:
        points = np.array(annotation).astype(np.float32)
        points *= scale
        points = points.flatten()
        adjusted_annotations.append(points)
    return adjusted_annotations


y_adjusted = adjust_labels_scaled(y)


def generate_heatmaps(annotations, image_size=IMAGE_SIZE, sigma=SIGMA):
    heatmaps = []
    for annotation in annotations:
        heatmap = np.zeros((image_size, image_size), dtype=np.float32)
        num_points = int(len(annotation) / 2)
        for i in range(num_points):
            idx = i * 2
            x, y = annotation[idx], annotation[idx + 1]
            if x == 0 and y == 0:
                continue
            if x >= image_size or y >= image_size or x < 0 or y < 0:
                continue
            xv, yv = np.meshgrid(np.arange(image_size), np.arange(image_size))
            heatmap += np.exp(-((xv - x) ** 2 + (yv - y)
                              ** 2) / (2 * sigma ** 2))
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        heatmaps.append(heatmap)
    heatmaps = np.array(heatmaps)
    heatmaps = np.expand_dims(heatmaps, axis=-1)
    return heatmaps


y_heatmaps = generate_heatmaps(y_adjusted)


def create_heatmap_model(image_size=IMAGE_SIZE):
    inputs = keras.Input(shape=(image_size, image_size, 3))

    x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    x = keras.layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    x = keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)

    outputs = keras.layers.Conv2D(
        1, (1, 1), activation='sigmoid', padding='same')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


from_pretrained_model = True

if from_pretrained_model:
    model = load_model('./models/v7.h5', compile=False)
else:
    model = create_heatmap_model()

optimizer = keras.optimizers.Adam(learning_rate=0.0001)
model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

early_stop = EarlyStopping(
    monitor='val_mae', patience=30, restore_best_weights=True)

reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_mae',
                                              factor=0.5,
                                              patience=5,
                                              min_lr=1e-7)

X_train, X_val, y_train, y_val = train_test_split(
    X, y_heatmaps, test_size=0.2, random_state=42)

batch_size = 8

train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
train_dataset = (
    train_dataset
    .shuffle(len(X_train))
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
val_dataset = (
    val_dataset
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

history = model.fit(
    train_dataset,
    epochs=1000,
    validation_data=val_dataset,
    callbacks=[early_stop, reduce_lr]
)

model.save('./models/v8.h5')
