import torch

USE_GPU = True
#Se recomienda utilizar gpu si se puede, se ha visto un aumento entre el 80 y 100% de velocidad entre cpu y gpu
if USE_GPU and torch.cuda.is_available():
    print("Cuda activado, se utilizara la gpu")
else:
    print("CUDA desactivado, el codigo se ejecutara en CPU")

def device():
    return torch.device("cuda:0" if (torch.cuda.is_available() and USE_GPU) else "cpu")