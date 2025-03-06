import os
from PIL import Image

def resize_image(image_path, output_size=(500, 500)):
    with Image.open(image_path) as img:
        img = img.resize(output_size)
        img.save(image_path)

def normalize_points(points_path, original_size, output_size=(500, 500)):
    with open(points_path, 'r') as file:
        points = file.readlines()

    original_width, original_height = original_size
    output_width, output_height = output_size

    normalized_points = []
    for point in points:
        x, y = map(float, point.strip().split(','))
        x = round((x / original_width) * output_width)
        y = round((y / original_height) * output_height)
        normalized_points.append(f"{x},{y}\n")

    with open(points_path, 'w') as file:
        file.writelines(normalized_points)

def process_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.endswith('.png'):
            image_path = os.path.join(folder_path, filename)
            points_path = os.path.join(folder_path, f"{os.path.splitext(filename)[0]}_points.txt")

            with Image.open(image_path) as img:
                original_size = img.size

            resize_image(image_path)
            normalize_points(points_path, original_size)

if __name__ == "__main__":
    folder_path = './dataset'
    process_folder(folder_path)