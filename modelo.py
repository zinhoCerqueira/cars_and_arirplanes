import tensorflow as tf
import numpy as np
from keras import datasets, layers, models

## Carregar o conjunto de dados CIFAR-10
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalizar os valores dos pixels para o intervalo [0, 1]
train_images, test_images = train_images / 255.0, test_images / 255.0

# Filtrar apenas as imagens de carro e avião
car_airplane_train_mask = np.squeeze((train_labels == 1) | (train_labels == 0))  # Ajustar a dimensão da máscara
car_airplane_test_mask = np.squeeze((test_labels == 1) | (test_labels == 0))  # Ajustar a dimensão da máscara
train_images_filtered, train_labels_filtered = train_images[car_airplane_train_mask], train_labels[car_airplane_train_mask]
test_images_filtered, test_labels_filtered = test_images[car_airplane_test_mask], test_labels[car_airplane_test_mask]

# Criar o modelo de rede neural
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(2)  # Duas classes: carro e avião
])

# Compilar o modelo
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Treinar o modelo
model.fit(train_images_filtered, train_labels_filtered, epochs=10, 
          validation_data=(test_images_filtered, test_labels_filtered))

# Avaliar a precisão do modelo
test_loss, test_acc = model.evaluate(test_images_filtered, test_labels_filtered, verbose=2)
print('\nTest accuracy:', test_acc)

model.save('cifar_car_airplane_model.h5')

