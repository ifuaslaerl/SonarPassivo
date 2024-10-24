""" Module made to train Artificial Neuron Networks. """
import torch
import torch.utils
from src import networks, mat_dataset

NE = 75
BS = 1 # tamanho dos conjuntos trabalhados

if __name__ == "__main__" :

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )

    print(f"Using {DEVICE} device")

    trainset = mat_dataset.MatDataset("data/Datasets/DadosSonar/train")
    validateset = mat_dataset.MatDataset("data/Datasets/DadosSonar/validate")

    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=BS,
                                            shuffle=True,
                                            num_workers=2)

    validateloader = torch.utils.data.DataLoader(validateset,
                                            batch_size=BS,
                                            shuffle=True,
                                            num_workers=2)

    classes = trainset.classes
    model = networks.SonarCNN(classes)
<<<<<<< HEAD
    train_data = network.fit(model, trainloader, validateloader, "data/Networks/robust", NE)
=======
    networks.fit(model, trainloader, validateloader, "data/Networks/robust", NE)
>>>>>>> 0f405ae50c0fec562440135a234378fe91c7a611
