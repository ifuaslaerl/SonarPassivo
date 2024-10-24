""" Module made to train Artificial Neuron Networks. """
import torch
import torch.utils
from src import networks, mat_dataset, data_organize

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

    trainset = mat_dataset.MatDataset("data/Datasets/DadosSonar/adversarial_training")
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
    train_data = networks.fit(model, trainloader, validateloader, "data/Networks/robust", NE)

    file_name = data_organize.find_name('data/Training_data/data','.csv')
    train_data.to_csv(file_name)
