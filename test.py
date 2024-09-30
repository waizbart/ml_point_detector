import numpy as np
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
import matplotlib.pyplot as plt

IMAGE_SIZE = 256
MAX_POINTS = 500

# Carregar o modelo
model = keras.models.load_model('hair_point_counter_model.h5')

# Função para carregar os dados de teste
def load_test_data(test_dataset_dir):
    images = []
    filenames = []
    
    for filename in os.listdir(test_dataset_dir):
        if filename.endswith('.png'):
            img_path = os.path.join(test_dataset_dir, filename)
            img = load_img(img_path, target_size=(IMAGE_SIZE, IMAGE_SIZE))
            img_array = img_to_array(img) / 255.0
            images.append(img_array)
            filenames.append(filename)
    
    return np.array(images), filenames

X_test, test_filenames = load_test_data('./dataset') 

# Fazer previsões
predictions = model.predict(X_test)

# Função para extrair pontos previstos
def extract_points(predictions): 
    points = []
    for pred in predictions:
        point_pairs = pred[:MAX_POINTS * 2].reshape(-1, 2)  
        points.append(point_pairs)
    return points

predicted_points = extract_points(predictions)

# Selecionar uma imagem aleatória do dataset
random_index = np.random.randint(len(X_test))
random_image = X_test[random_index]
predicted_points_random = predicted_points[random_index]
filename_random = test_filenames[random_index]

# Mostrar a imagem com os pontos previstos
plt.figure(figsize=(8, 8))
plt.imshow(random_image)
plt.scatter(predicted_points_random[:, 0] * IMAGE_SIZE, predicted_points_random[:, 1] * IMAGE_SIZE, color='red', s=10)  # Redimensionando os pontos
plt.title(f'Predicted points for {filename_random}')
plt.axis('off')
plt.show()
