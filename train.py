import os
from matplotlib import pyplot as plt
import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import backend as K

MAX_POINTS = 100
IMAGE_SIZE = 256
SIGMA = 4

def load_data(dataset_dir):
    images = []
    points = []

    for filename in os.listdir(dataset_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(dataset_dir, filename)
            annotation_path = os.path.join(dataset_dir, filename.replace('.png', '_points.txt'))

            img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)

            with open(annotation_path, 'r') as f:
                coords = [list(map(float, line.strip().split(','))) for line in f.readlines()]
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
            heatmap += np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma ** 2))
        heatmap = heatmap / np.max(heatmap) if np.max(heatmap) > 0 else heatmap
        heatmaps.append(heatmap)
    heatmaps = np.array(heatmaps)
    heatmaps = np.expand_dims(heatmaps, axis=-1)
    return heatmaps

y_heatmaps = generate_heatmaps(y_adjusted)

def visualize_heatmap_example(X, y_heatmaps, index=0):
    image = X[index]
    heatmap = y_heatmaps[index, :, :, 0] 

    plt.figure(figsize=(15, 5))

    # Imagem Original
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title('Imagem Original')
    plt.axis('off')

    # Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title('Heatmap')
    plt.axis('off')

    # Sobreposição
    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(heatmap, cmap='jet', alpha=0.5)
    plt.title('Sobreposição')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

visualize_heatmap_example(X, y_heatmaps, index=0)

def create_heatmap_model(image_size=IMAGE_SIZE):
    inputs = keras.Input(shape=(image_size, image_size, 3))

    x = keras.layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(64, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(128, (3, 3), padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.LeakyReLU(alpha=0.1)(x)
    x = keras.layers.Dropout(0.2)(x)
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

    outputs = keras.layers.Conv2D(1, (1, 1), activation='sigmoid', padding='same')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_heatmap_model()

optimizer = keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='mae', metrics=['mae'])

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_mae', patience=50, restore_best_weights=True)

history = model.fit(
    X,
    y_heatmaps,
    epochs=1000,
    batch_size=8,
    validation_split=0.2,
    callbacks=[early_stop]
)

model.save('./models/v7.h5')
