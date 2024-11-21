import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MAX_POINTS = 500
IMAGE_SIZE = 256
SIGMA = 2  

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

def adjust_labels_normalized(annotations, image_size=IMAGE_SIZE):
    adjusted_annotations = []
    for annotation in annotations:
        points = np.array(annotation).astype(np.float32)
        points /= image_size  # Normaliza para [0, 1]
        points = points.flatten()
        if len(points) < MAX_POINTS * 2:
            points = np.pad(points, (0, MAX_POINTS * 2 - len(points)), 'constant')
        adjusted_annotations.append(points[:MAX_POINTS * 2])
    return np.array(adjusted_annotations)

y_adjusted = adjust_labels_normalized(y)

def generate_heatmaps(annotations, image_size=IMAGE_SIZE, sigma=SIGMA):
    heatmaps = []
    for annotation in annotations:
        heatmap = np.zeros((image_size, image_size, MAX_POINTS), dtype=np.float32)
        for i in range(MAX_POINTS):
            idx = i * 2
            if idx + 1 >= len(annotation):
                continue
            x_norm, y_norm = annotation[idx], annotation[idx + 1]
            if x_norm == 0 and y_norm == 0:
                continue  # Ponto ausente
            x = int(x_norm * image_size)
            y = int(y_norm * image_size)
            if x >= image_size or y >= image_size:
                continue  # Coordenadas fora da imagem
            # Cria uma Gaussiana no ponto
            xv, yv = np.meshgrid(np.arange(image_size), np.arange(image_size))
            heatmap[..., i] += np.exp(-((xv - x) ** 2 + (yv - y) ** 2) / (2 * sigma ** 2))
        heatmaps.append(heatmap)
    return np.array(heatmaps)

y_heatmaps = generate_heatmaps(y_adjusted)

print("Shape das imagens X:", X.shape)
print("Shape dos heatmaps y:", y_heatmaps.shape)

def create_heatmap_model(image_size=IMAGE_SIZE, num_points=MAX_POINTS):
    inputs = keras.Input(shape=(image_size, image_size, 3))

    x = keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = keras.layers.MaxPooling2D((2, 2))(x)

    x = keras.layers.Conv2DTranspose(128, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(64, (3, 3), strides=2, padding='same', activation='relu')(x)
    x = keras.layers.Conv2DTranspose(32, (3, 3), strides=2, padding='same', activation='relu')(x)

    outputs = keras.layers.Conv2D(num_points, (1, 1), activation='sigmoid', padding='same')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

model = create_heatmap_model()

model.compile(optimizer='adam', loss='binary_crossentropy')

model.summary()

from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(
    X,
    y_heatmaps,
    epochs=100,
    batch_size=16,
    validation_split=0.2,
    callbacks=[early_stop]
)

model.save('hair_point_counter_model.h5')

def get_predicted_points(heatmaps):
    predicted_points = []
    for heatmap in heatmaps:
        points = []
        for i in range(MAX_POINTS):
            heatmap_i = heatmap[..., i]
            y, x = np.unravel_index(np.argmax(heatmap_i), heatmap_i.shape)
            points.extend([x / IMAGE_SIZE, y / IMAGE_SIZE])  
        predicted_points.append(points)
    return np.array(predicted_points)

def predict_points(image_path):
    img = load_img(image_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0) 

    predicted_heatmaps = model.predict(img_array)

    predicted_points = get_predicted_points(predicted_heatmaps)

    return predicted_points
