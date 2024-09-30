import os
import numpy as np
from PIL import Image, ImageDraw

# Parâmetros do dataset
num_images = 100  # Número de imagens a serem geradas
image_size = 256  # Tamanho da imagem (X por X)
max_points = 100    # Número máximo de pontos por imagem
point_radius = 5   # Raio do ponto

# Diretório para salvar o dataset
output_dir = './dataset'
os.makedirs(output_dir, exist_ok=True)

for i in range(num_images):
    # Criar uma imagem em branco
    img = Image.new('RGB', (image_size, image_size), color='white')
    draw = ImageDraw.Draw(img)
    
    # Gerar um número aleatório de pontos
    num_points = np.random.randint(1, max_points + 1)
    points = []

    for _ in range(num_points):
        # Gerar coordenadas aleatórias para os pontos
        x = np.random.randint(point_radius, image_size - point_radius)
        y = np.random.randint(point_radius, image_size - point_radius)
        points.append((x, y))

        # Desenhar um ponto na imagem (círculo)
        draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill='black')

    # Salvar a imagem
    img_path = os.path.join(output_dir, f'image_{i}.png')
    img.save(img_path)

    # Salvar as coordenadas em um arquivo de anotações
    annotation_path = os.path.join(output_dir, f'image_{i}_points.txt')
    with open(annotation_path, 'w') as f:
        for point in points:
            f.write(f"{point[0]},{point[1]}\n")  # Escreve x,y
