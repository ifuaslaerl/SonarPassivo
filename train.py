#Bibliotecas

import torch
import torchvision.transforms as transforms
import os
from src import Networks , MatDataset

NE = 75
BS = 1 # tamanho dos conjuntos trabalhados
data = 'DadosSonar'
root = ""

def main():

    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
    
    print(f'Using {device} device')

    trainset = MatDataset.MatDataset(f'{data}/train')

    validateset = MatDataset.MatDataset(f'{data}/validate')

    trainloader = torch.utils.data.DataLoader(trainset,
                                        batch_size=BS,
                                        shuffle=True,
                                        num_workers=2)

    validateloader = torch.utils.data.DataLoader(validateset,
                                        batch_size=BS,
                                        shuffle=True,
                                        num_workers=2)
    
    classes = trainset.classes

    model = Networks.Sonar_CNN(classes)

    Networks.fit(model,trainloader,validateloader,root,NE)

if __name__ == "__main__" : main()
