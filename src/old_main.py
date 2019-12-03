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
dir_imagenes = 'train/'
# cant_archivos: Cuantos archivos del repositorio usar.
# El valor 0 significa usar todos.
# Se puede poner un número arbitrario para pruebas
cant_archivos = 0
# ruta_trainlabels: Ruta del archivo trainLabels.csv (relativa a donde está este .py)
ruta_trainlabels = 'trainLabels.csv'


# Constructor para el Dataset basado en las imágenes
class Cifar10Dataset(torch.utils.data.Dataset):
    # data_dir: El directorio del que se leerán las imágenes
    # label_source: De dónde se obtendrán las etiquetas
    # data_size: Cuantos archivos usar (0 = todos)
    def __init__(self, data_dir, label_source, data_size=0):
        files = os.listdir(data_dir)
        files = [os.path.join(data_dir, x) for x in files]
        if data_size < 0 or data_size > len(files):
            assert "Data size should be between 0 to number of files in the dataset"
        if data_size == 0:
            data_size = len(files)
        self.data_size = data_size
        self.files = random.sample(files, self.data_size)
        self.label_source = label_source

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_address = self.files[idx]
        image = np.array(Image.open(image_address))
        # Se deja los valores de la imágen en el rango 0-1
        image = image / 255
        # Se traspone la imagen para que el canal sea la primer coordenada
        # (la red espera NxMx3)
        image = image.transpose(2, 0, 1)
        image = torch.Tensor(image)
        # Se puede agregar: Aplicar normalización (Hacer que los valores vayan
        # entre -1 y 1 pero con el 0 en el valor promedio.
        # Los parámetros estos están precalculados para el set CIFAR-10
        # image = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(image)
        label_idx = int(image_address[:-4].split("/")[1]) - 1
        label = self.label_source[label_idx]
        label = torch.tensor(label).long()
        return image, label


# Levantamos los labels del archivo csv
labels = pd.read_csv(ruta_trainlabels)
# Lo transformamos a números con un labelEncoder
# labels_encoder es importante: Es el que me va a permitir revertir la
# transformación para conocer el nombre de una etiqueta numérica
labels_encoder = LabelEncoder()
labels_numeros = labels_encoder.fit_transform(labels['label'])

# Generamos el DataSet con nuestros datos de entrenamiento
cifar_dataset = Cifar10Dataset(data_dir=dir_imagenes, data_size=cant_archivos, label_source=labels_numeros)


##Antes de pasar a la separación en datos de training y test, podemos verificar
##que estamos levantando las imágenes de manera correcta. Defino una función que
##dado un número toma la imágen en esa posición del dataset (Ojo, recordar que
##está mezclado), y grafica la imágen junto con su etiqueta.
def mostrarImagen(dataset, nroImagen, encoder):
    imagen, etiqueta = dataset[nroImagen]
    # Se regresa la imágen a formato numpy
    # Es necesario trasponer la imágen para que funcione con imshow
    # (imshow espera 3xNxM)
    imagen = imagen.numpy()
    imagen = imagen.transpose(1, 2, 0)
    plt.imshow(imagen)
    # Recupero la etiqueta de la imágen usando el encoder
    plt.title(labels_encoder.inverse_transform([etiqueta])[0])


# ej: mostrarImagen(cifar_dataset, 10, labels_encoder)


# batch_size = Cuántos archivos entran por batch de entrenamiento
# (Nota: En una epoch todos los archivos terminan pasando, pero la
#       corrección de los pesos y parámetros se hace cada batch)
batch_size = 84

# Proporción de archivos a usar para test
test_proportion = .2
train_size = int((1 - test_proportion) * len(cifar_dataset))
test_size = len(cifar_dataset) - train_size

# Creo los Datasets y los loaders que voy a utilizar para el aprendizaje
train_dataset, test_dataset = torch.utils.data.random_split(cifar_dataset, [train_size, test_size])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)


# Creo la red (Es la misma que usamos en clase)
class CifarNet(nn.Module):
    def __init__(self):
        super(CifarNet, self).__init__()
        # Capa convolución 1: Toma imágenes con 3 canales y un kernel de
        # 5x5 para generar imágenes de 32 canales
        # Entrada: 32x32; Salida: 28x28 (Se pierden 2 pixeles de cada borde)
        self.conv1 = nn.Conv2d(3, 32, 5)
        # Capa MaxPool. Se deja un elemento de cada kernel de 2x2
        # Entrada: 28x28; Salida: 14x14
        self.pool = nn.MaxPool2d(2, 2)
        # Capa convolución 2: Toma imágenes de 32 canales y un kernel de
        # 5x5 para generar imágenes de 16 canales.
        # Entrada: 14x14; Salida: 10x10
        self.conv2 = nn.Conv2d(32, 16, 5)
        # Luego de conv2 se hace otro MaxPool, así que
        # Entrada: 10x10; Salida: 5x5
        # Finalmente llegamos a la red lineal, como teníamos 16 canales
        # y una imágen de 5x5, la cantidad de entradas de la capa es
        # 16x5x5.
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x


model = CifarNet()

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

    # Hacemos el train con los datos que salen del loader
    train_loss = train(model, train_loader, optimizer)

    # Probamos el nuevo entrenamiento sobre los datos de test
    test_loss, accuracy = test(model, test_loader)

    # Guardamos en nuestras listas los datos de loss obtenidos
    epoch_nums.append(epoch)
    training_loss.append(train_loss)
    validation_loss.append(test_loss)

    # Cada 10 iteraciones vamos imprimiendo nuestros resultados parciales
    if (epoch) % 10 == 0:
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

