import os
import random
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

max_width = 224
max_height = 224

from dataset import Dataset, mostrarImagen


# Generamos el DataSet con nuestros datos de entrenamiento
dataset = Dataset(data_dir=dir_imagenes, max_width=max_width, max_height=max_height, data_size=cant_archivos)

# ej: mostrarImagen(cifar_dataset, 10, labels_encoder)


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
            test_loss += loss_criteria(out, target).item()

            # Calculamos la accuracy (exactitud) (Sumando el resultado como
            # correcto si la predicción acertó)
            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target == predicted).item()

    # Devolvemos la exactitud y loss promedio
    avg_accuracy = correct / len(data_loader.dataset)
    avg_loss = test_loss / len(data_loader.dataset)
    return avg_loss, avg_accuracy


# Definimos nuestro criterio de loss
# Aquí usamos CrossEntropyLoss, que está poensado para clasificación
loss_criteria = nn.CrossEntropyLoss()

# Definimos nuestro optimizer
# Aquí usamos Stochastic Gradient Descent (SGD) - Descenso por Gradiente Estocástico
learning_rate = 0.01
learning_momentum = 0.9
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=learning_momentum)

# En estas listas vacías nos vamos guardando el loss para los datos de training
# y validación en cada iteración.
epoch_nums = []
training_loss = []
validation_loss = []

# Entrenamiento. Por default lo hacemos por 100 iteraciones (epochs)
epochs = 100
for epoch in range(1, epochs + 1):

    print("Empiezo a entrenar")
    # Hacemos el train con los datos que salen del loader
    train_loss = train(model, train_loader, optimizer)

    print("Empiezo a testear")
    # Probamos el nuevo entrenamiento sobre los datos de test
    test_loss, accuracy = test(model, test_loader)

    print("Termine de entrenar y testear")
    # Guardamos en nuestras listas los datos de loss obtenidos
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)

    # Cada 10 iteraciones vamos imprimiendo nuestros resultados parciales
    #if (epoch) % 10 == 0:
    print('Epoch {:d}: loss entrenamiento= {:.4f}, loss validacion= {:.4f}, exactitud={:.4%}'.format(epoch,
                                                                                                         train_loss,
                                                                                                         test_loss,
                                                                                                         accuracy))

# Creamos la matriz de confusión, esta es parte del paquete scikit
from sklearn.metrics import confusion_matrix

# Ponemos el modelo en modo evaluación
model.eval()

# Hacemos las predicciones para los datos de test
# Para eso, en primer lugar generamos la matriz de entradas y vector de
# resultados a partir del dataloader
entradas = list()
salidas = list()
for batch, tensor in enumerate(test_loader):
    valor, salida = tensor
    entradas.append(valor)
    salidas.append(salida)
# Se pasan a formato Tensor
entradas = torch.cat(entradas)
salidas = torch.cat(salidas)
# Se obtienen las predicciones
_, predicted = torch.max(model(entradas), 1)

# Graficamos la matriz de confusión
cm = confusion_matrix(salidas.numpy(), predicted.numpy())
plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
plt.colorbar()
tick_marks = np.arange(10)
plt.xticks(tick_marks, labels_encoder.inverse_transform(range(10)), rotation=45)
plt.yticks(tick_marks, labels_encoder.inverse_transform(range(10)))
plt.xlabel("El modelo predijo que era")
plt.ylabel("La imágen real era")
plt.show()

