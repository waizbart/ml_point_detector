import os

dataset_dir = './dataset'

for filename in os.listdir(dataset_dir):
    if filename.endswith('.png'):
        img_path = os.path.join(dataset_dir, filename)
        annotation_path = os.path.join(dataset_dir, filename.replace('.png', '_points.txt'))
        
        if not os.path.isfile(annotation_path):
            os.remove(img_path)
