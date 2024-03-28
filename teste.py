import tensorflow as tf
import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt
import os

# Carregar o modelo treinado
model = tf.keras.models.load_model('cifar_car_airplane_model.h5')

# Pasta contendo todas as imagens
img_folder = './img/'

# Listas para armazenar os caminhos das imagens e as previsões
img_paths = []
predictions_list = []

# Iterar sobre cada arquivo na pasta de imagens
for root, dirs, files in os.walk(img_folder):
    for file in files:
        # Caminho completo da imagem
        img_path = os.path.join(root, file)
        img_paths.append(img_path)

        # Carregar e pré-processar a imagem
        img = image.load_img(img_path, target_size=(32, 32))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)
        img_array = img_array / 255.0  # Normalizar os valores dos pixels

        # Fazer previsões na imagem
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions[0])

        # Interpretar as previsões
        class_names = ['avião', 'carro']
        predicted_class_name = class_names[predicted_class]
        predictions_list.append(predicted_class_name)

# Criar uma figura para exibir as imagens e as previsões
plt.figure(figsize=(15, 10))

# Iterar sobre cada imagem e sua previsão
for i, (img_path, prediction) in enumerate(zip(img_paths, predictions_list)):
    # Carregar a imagem
    img = image.load_img(img_path, target_size=(150, 150))

    # Adicionar a imagem ao subplot
    plt.subplot(4, 4, i + 1)
    plt.imshow(img)
    plt.axis('off')  # Ocultar eixos
    plt.title(prediction)  # Adicionar a previsão como título

plt.tight_layout()  # Ajustar layout
plt.show()
