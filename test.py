""" App made to test Networks. """

import os
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from SonarPassivo.src import networks
from src import MatDataset

NE = 75
BS = 1 # tamanho dos conjuntos trabalhados

def show(matriz,title):
    plt.title(title)
    sns.heatmap(matriz, annot=True, fmt="d", cmap="cividis", cbar=True, annot_kws={"size" : 16})
    plt.show()

if __name__ == "__main__" :
    DATA = "data/DadosSonar"
    MODELS = "data/Networks"

    DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu" )

    print(f'Using {DEVICE} device')

    testset = MatDataset.MatDataset(f'{DATA}/test')

    testloader = torch.utils.data.DataLoader(testset,
                                        batch_size=BS,
                                        shuffle=True,
                                        num_workers=2)

    model = networks.Sonar_CNN(testset.classes)

    for archive in os.listdir(MODELS):
        check = torch.load(os.path.join(MODELS,archive))
        model.load_state_dict(check['model_state_dict'])
        model.optimizer.load_state_dict(check['optimizer_state_dict'])
        loss = check['loss_out']
        accuracy = check['accuracy']
        model.eval()
        real_loss , real_accuracy , matrix = networks.test_loop(model,testloader)
        print(f'{archive} - loss_out = {loss :.3f} - real_loss = {real_loss :.3f} - accuracy = {accuracy :.3f} - real_accuracy = {real_accuracy :.3f}')
        show(matrix,f'{real_loss}')
