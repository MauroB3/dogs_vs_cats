import math
import os
import random
import time
from torch.nn.utils import clip_grad_norm_, clip_grad_value_

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
from network2 import Network2

model = Network()

input('Listo para entrenar')


def train(model, data_loader, optimizer):

    model.train()
    train_loss = 0

    for batch, tensor in enumerate(data_loader):
        data, target = tensor

        optimizer.zero_grad()
        out = model(data)
        loss = loss_criteria(out, target)
        train_loss += loss.item()

        loss.backward()
        ###

        # Clip the gradients norm to avoid them becoming too large
        clip_grad_norm_(model.parameters(), 5)

        # Update the LR
        optimizer.step()
        scheduler.step()

    avg_loss = train_loss / len(data_loader.dataset)
    return avg_loss



def test(model, data_loader):

    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for batch, tensor in enumerate(data_loader):
            data, target = tensor

            out = model(data)


            test_loss += loss_criteria(out, target).to(device()).item()


            _, predicted = torch.max(out.data, 1)
            correct += torch.sum(target == predicted).to(device()).item()

    # Devolvemos la exactitud y loss promedio
    avg_accuracy = correct / len(data_loader.dataset)
    avg_loss = test_loss / len(data_loader.dataset)
    return avg_loss, avg_accuracy

def cyclical_lr(stepsize, min_lr=3e-2, max_lr=3e-3):

    # Scaler: we can adapt this if we do not want the triangular CLR
    scaler = lambda x: 1.

    # Lambda function to calculate the LR
    lr_lambda = lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize)

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x)) * scaler(cycle)

    return lr_lambda


# Definimos nuestro criterio de loss
# Aquí usamos CrossEntropyLoss, que está poensado para clasificación
loss_criteria = nn.CrossEntropyLoss()

# Definimos nuestro optimizer
# Aquí usamos Stochastic Gradient Descent (SGD) - Descenso por Gradiente Estocástico

factor = 6
learning_rate = 0.01
learning_momentum = 0.9
lr_find_epochs = 2
start_lr = 1e-7
end_lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=1.)
step_size = 4*len(train_loader)
clr = cyclical_lr(step_size, min_lr=end_lr/factor, max_lr=end_lr)
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [clr])


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


import math

dir_imagenes = '../eval_data/'
data_ds = Dataset(data_dir=dir_imagenes, max_width=max_width, max_height=max_height, data_size=cant_archivos)
data_ld = torch.utils.data.DataLoader(data_ds, batch_size=batch_size, shuffle=False, num_workers=0)


def evaluate_data(model, data_loader):
    model.eval()
    result = []

    with torch.no_grad():
        for batch, tensor in enumerate(data_loader):
            data, target = tensor
            out = model(data)
            batch_results = torch.softmax(out, dim=1)[:, 1].tolist()
            result += batch_results
    return {k: math.ceil(v) for k, v in enumerate(result)}

result = evaluate_data(model, data_ld)

result_pd = pd.DataFrame({'id': list(result.keys()), 'label': list(result.values())})
result_pd.to_csv('result.csv', index=False)
