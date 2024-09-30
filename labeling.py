import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import os
import uuid

# Função para salvar os pontos (x, y) em um arquivo
def save_points(points, random_name):
    dataset_dir = "./dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Cria o arquivo para salvar os pontos
    filename = os.path.join(dataset_dir, random_name + "_points.txt")
    with open(filename, "w") as file:
        for point in points:
            file.write(f"{point[0]}, {point[1]}\n")
    print(f"Pontos salvos em: {filename}")

# Função para salvar a imagem no diretório /dataset com nome aleatório
def save_image(img, image_path):
    dataset_dir = "./dataset"
    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    # Gera um nome aleatório
    random_name = str(uuid.uuid4())
    save_path = os.path.join(dataset_dir, random_name + ".png")

    # Salva a imagem no formato PNG
    img.save(save_path)
    print(f"Imagem salva em: {save_path}")
    return random_name

# Função para desenhar uma cruz na posição (x, y)
def draw_cross(x, y, canvas):
    cross_size = 5
    # Linha horizontal da cruz
    canvas.create_line(x - cross_size, y, x + cross_size, y, fill="red", width=2)
    # Linha vertical da cruz
    canvas.create_line(x, y - cross_size, x, y + cross_size, fill="red", width=2)

# Função para remover a cruz na posição (x, y)
def remove_cross(x, y, canvas):
    items = canvas.find_all()
    for item in items:
        coords = canvas.coords(item)
        # Remover cruz se o centro estiver próximo da posição clicada
        if len(coords) == 4 and abs((coords[0] + coords[2]) / 2 - x) < 5 and abs((coords[1] + coords[3]) / 2 - y) < 5:
            canvas.delete(item)

# Função para encontrar e remover um ponto próximo ao clique
def remove_point(event, points, canvas):
    x, y = event.x, event.y
    # Tolerância para considerar um ponto próximo
    tolerance = 5
    for point in points:
        if abs(point[0] - x) < tolerance and abs(point[1] - y) < tolerance:
            points.remove(point)
            print(f"Ponto removido: ({point[0]}, {point[1]})")
            # Remove a cruz associada ao ponto
            remove_cross(point[0], point[1], canvas)
            break

# Função para manipular o clique na imagem
def on_click(event, points, random_name, canvas):
    if event.num == 1:  # Clique esquerdo
        x, y = event.x, event.y
        points.append((x, y))
        print(f"Você clicou na posição: ({x}, {y})")

        # Desenha a cruz no ponto clicado
        draw_cross(x, y, canvas)

        # Salva os pontos sempre que um novo é adicionado
        save_points(points, random_name)
    elif event.num == 3:  # Clique direito
        remove_point(event, points, canvas)
        save_points(points, random_name)  # Atualiza o arquivo após remover o ponto

# Função para selecionar a imagem
def select_image():
    image_path = filedialog.askopenfilename(title="Selecione uma imagem", filetypes=[("Imagens", "*.png;*.jpg;*.jpeg;*.bmp")])
    if not image_path:
        return
    
    # Abre e redimensiona a imagem
    img = Image.open(image_path)
    img = img.resize((500, 500))  # Ajusta a imagem para caber na tela
    img_tk = ImageTk.PhotoImage(img)

    # Salva a imagem no diretório /dataset com um nome aleatório
    random_name = save_image(img, image_path)

    # Exibe a imagem no canvas
    canvas.config(width=500, height=500)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk

    # Inicializa a lista de pontos
    points = []

    # Configura o evento de clique na imagem
    canvas.bind("<Button-1>", lambda event: on_click(event, points, random_name, canvas))  # Clique esquerdo
    canvas.bind("<Button-3>", lambda event: on_click(event, points, random_name, canvas))  # Clique direito

# Cria a janela principal
root = tk.Tk()
root.title("Seleção de Pontos na Imagem")

# Cria o botão de seleção de imagem
btn_select = tk.Button(root, text="Selecionar Imagem", command=select_image)
btn_select.pack(pady=10)

# Cria um canvas para exibir a imagem
canvas = tk.Canvas(root)
canvas.pack()

# Inicia o loop da interface
root.mainloop()
