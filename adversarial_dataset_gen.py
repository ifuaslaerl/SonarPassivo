""" Module made to Debug Artificial Neuron Networks Libraries. """
import os
import torch
import torch.utils
from src import networks , mat_dataset

NE = 75
BS = 1 # tamanho dos conjuntos trabalhados

if __name__ == "__main__" :

    DATA = "data/Datasets/DadosSonar"
    ROOT = ""

    DEVICE = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
        )

    print(f"Using {DEVICE} device")

    trainset = mat_dataset.MatDataset(f"{DATA}/train")
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=BS,
                                            shuffle=True,
                                            num_workers=2)

    classes = trainset.classes
    model = networks.SonarCNN(classes)

    # por um for aq depois
    NETWORK = "0976.pth"
    check = torch.load(os.path.join("data/Networks",NETWORK))
    model.load_state_dict(check['model_state_dict'])
    model.optimizer.load_state_dict(check['optimizer_state_dict'])
    loss = check['loss_out']
    accuracy = check['accuracy']

    model.eval()
    data = networks.adv_data_gen(model, trainloader, 1e-6)

    mat_dataset.save_set(data,"data/Datasets/DadosSonar/adversarial",classes)
