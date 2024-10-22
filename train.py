""" Module made to train Artificial Neuron Networks. """
import torch
import torch.utils
from SonarPassivo.src import networks
from src import MatDataset

NE = 75
BS = 1 # tamanho dos conjuntos trabalhados

if __name__ == "__main__" :

    DATA = "data/DadosSonar"
    ROOT = ""

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )

    print(f"Using {DEVICE} device")

    trainset = MatDataset.MatDataset(f"{DATA}/train")
    validateset = MatDataset.MatDataset(f"{DATA}/validate")

    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=BS,
                                            shuffle=True,
                                            num_workers=2)

    validateloader = torch.utils.data.DataLoader(validateset,
                                            batch_size=BS,
                                            shuffle=True,
                                            num_workers=2)

    classes = trainset.classes
    model = networks.Sonar_CNN(classes)
    networks.fit(model, trainloader, validateloader, ROOT, NE)
