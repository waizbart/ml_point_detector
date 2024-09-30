import os
import numpy as np
from tensorflow import keras
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

MAX_POINTS = 500
IMAGE_SIZE = 256

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
                coords = [list(map(int, line.strip().split(','))) for line in f.readlines()]
                points.append(coords)

    return np.array(images), points

X, y = load_data('./dataset')

def adjust_labels(annotations):
    adjusted_annotations = []
    for annotation in annotations:
        points = np.array(annotation).flatten()  
        if len(points) < MAX_POINTS * 2: 
            points = np.pad(points, (0, MAX_POINTS * 2 - len(points)), 'constant')
        adjusted_annotations.append(points[:MAX_POINTS * 2]) 
    return np.array(adjusted_annotations)

y_adjusted = adjust_labels(y)

print("shapes", X.shape)

def create_model():
    model = keras.models.Sequential()

    model.add(keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3)))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Conv2D(256, (3, 3), activation='relu'))
    model.add(keras.layers.MaxPooling2D((2, 2)))

    model.add(keras.layers.Flatten())

    model.add(keras.layers.Dense(512, activation='relu'))
    
    model.add(keras.layers.Dense(MAX_POINTS * 2, activation='linear'))

    return model

model = create_model()
model.compile(optimizer='adam', loss=keras.losses.MeanSquaredError(), metrics=['accuracy'])
model.summary()

history = model.fit(X, y_adjusted, epochs=500, batch_size=16, validation_split=0.2)

model.save('hair_point_counter_model.h5')

