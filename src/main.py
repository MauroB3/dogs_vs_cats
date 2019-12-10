import os
import random
import time

import pandas as pd
import PIL.Image as Image
import numpy as np
import matplotlib.pyplot as plt
import torch.utils.data.dataset
from sklearn.preprocessing import LabelEncoder
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler

# Algunos parámetros:
# dir_imagenes: Directorio donde están las imágenes extraídas del 7zip de Kaggle
# (Las que valen son las de TRAIN, que son aquellas que tienen las etiquetas en
# el archivo trainLabels.csv)
dir_imagenes = '../train/'
# cant_archivos: Cuantos archivos del repositorio usar.
# El valor 0 significa usar todos.
# Se puede poner un número arbitrario para pruebas
cant_archivos = 0

max_width = 200
max_height = 200

from config import device

from dataset import Dataset, mostrarImagen


# Generamos el DataSet con nuestros datos de entrenamiento
dataset = Dataset(data_dir=dir_imagenes, max_width=max_width, max_height=max_height, data_size=cant_archivos)

# batch_size = Cuántos archivos entran por batch de entrenamiento
# (Nota: En una epoch todos los archivos terminan pasando, pero la
#       corrección de los pesos y parámetros se hace cada batch)
batch_size = 8

# Proporción de archivos a usar para test
test_proportion = .2
train_size = int((1 - test_proportion) * len(dataset))
test_size = len(dataset) - train_size

# Creo los Datasets y los loaders que voy a utilizar para el aprendizaje
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

from network import Network

model = Network()

input('Listo para entrenar')


# Función que modela el entrenamiento de la red en cada epoch
def train(model, data_loader, optimizer):
    # El modelo se debe poner en modo training
    model.train()
    train_loss = 0

    for batch, tensor in enumerate(data_loader):
        data, target = tensor
        # Se pasan los datos por la red y se calcula la función de loss
        optimizer.zero_grad()
        out = model(data)
        loss = loss_criteria(out, target)
        train_loss += loss.item()

        # Se hace la backpropagation y se actualizan los parámetros de la red
        loss.backward()
        optimizer.step()

    # Se devuelve el loss promedio
    avg_loss = train_loss / len(data_loader.dataset)
    return avg_loss


# Función que realiza el test de la red en cada epoch
def test(model, data_loader):
    # Ahora ponemos el modelo en modo evaluación
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch, tensor in enumerate(data_loader):
            data, target = tensor
            # Dado el dato, obtenemos la predicción
            out = model(data)

            # Calculamos el loss
            test_loss += loss_criteria(out, target).to(device()).item()

            # Calculamos la accuracy (exactitud) (Sumando el resultado como
            # correcto si la predicción acertó)
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target == predicted).to(device()).item()

    # Devolvemos la exactitud y loss promedio
    avg_accuracy = correct / len(data_loader.dataset)
    avg_loss = test_loss / len(data_loader.dataset)
    return avg_loss, avg_accuracy


# Definimos nuestro criterio de loss
# Aquí usamos CrossEntropyLoss, que está poensado para clasificación
loss_criteria = nn.CrossEntropyLoss().to(device())

# Definimos nuestro optimizer
# Aquí usamos Stochastic Gradient Descent (SGD) - Descenso por Gradiente Estocástico
learning_rate = 0.01
learning_momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=learning_momentum)

epoch_nums = []
training_loss = []
validation_loss = []

epochs = 5
print("comenzando entrenamiento")
for epoch in range(1, epochs + 1):

    start = time.time()

    # Hacemos el train con los datos que salen del loader
    train_loss = train(model, train_loader, optimizer)

    # Probamos el nuevo entrenamiento sobre los datos de test
    test_loss, accuracy = test(model, test_loader)

    # Guardamos en nuestras listas los datos de loss obtenidos
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)
    end = time.time() - start

    print('Epoch {:d}: loss entrenamiento= {:.4f}, loss validacion= {:.4f}, exactitud={:.4%}, tiempo requerido={:.4f}'.format(epoch,
                                                                                                         train_loss,
                                                                                                         test_loss,
                                                                                                         accuracy, end))





