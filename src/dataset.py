# pad = transforms.Pad(50, padding_mode='reflect / constant')

import os
import random
import PIL.Image as Image
import numpy as np
import torch.utils.data.dataset
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

class Dataset(torch.utils.data.Dataset):
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
        self.max_width = 300
        self.max_height = 300

    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        image_address = self.files[idx]
        image = Image.open(image_address)
        self.resize_image(image)
        pad = transforms.Pad(50, padding_mode='constant')
        image = pad(image)
        image = np.array(image)
        # Se deja los valores de la imágen en el rango 0-1
        image = image / 255
        # Se traspone la imagen para que el canal sea la primer coordenada
        # (la red espera NxMx3)
        image = image.transpose(2, 0, 1)
        # Se puede agregar: Aplicar normalización (Hacer que los valores vayan
        # entre -1 y 1 pero con el 0 en el valor promedio.
        # Los parámetros estos están precalculados para el set CIFAR-10
        # image = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))(image)
        label = image_address[9:12]
        return image, label

    def resize_image(self, image):
        width, height = image.size
        padding_width = max(0, self.max_width - width)
        padding_height = max(0, self.max_height - height)
        pad = transforms.Pad((0, 0, padding_width, padding_height), padding_mode='constant')
        if(width > self.max_width and height > self.max_height):



def mostrarImagen(image, label):
    #imagen, etiqueta = dataset[nroImagen]
    # Se regresa la imágen a formato numpy
    # Es necesario trasponer la imágen para que funcione con imshow
    # (imshow espera 3xNxM)
    imagen = image.transpose(1, 2, 0)
    plt.imshow(imagen)
    # Recupero la etiqueta de la imágen usando el encoder
    plt.title(label)
    plt.show()

dataset = Dataset(data_dir='../train/', data_size=25000, label_source=2)

image, label = dataset.__getitem__(4)

mostrarImagen(image, label)
