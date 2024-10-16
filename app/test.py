#Bibliotecas

import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import MatDataset
import Networks
import seaborn as sns

NE = 75
BS = 1 # tamanho dos conjuntos trabalhados
data = 'DadosSonar'
models_ = "Networks"
root = ""

eps = 1e-6

def show(matriz,title):
    plt.title(title)
    sns.heatmap(matriz, annot=True, fmt="d", cmap="cividis", cbar=True, annot_kws={"size" : 16})
    plt.show()
    
def main():

    device = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu"
            )
    
    print(f'Using {device} device')

    testset = MatDataset.MatDataset(f'{data}/test')

    testloader = torch.utils.data.DataLoader(testset,
                                        batch_size=BS,
                                        shuffle=True,
                                        num_workers=2)

    model = Networks.Sonar_CNN(testset.classes)

    models = os.path.join(root,models_)
    for archive in os.listdir(models):
        
        check = torch.load(os.path.join(models,archive))
        
        model.load_state_dict(check['model_state_dict'])
        model.optimizer.load_state_dict(check['optimizer_state_dict'])
        loss = check['loss_out']
        accuracy = check['accuracy']

        model.eval()
        
        real_loss , real_accuracy , matrix = Networks.adv_test_loop(model,testloader,1e-3)
        
        print(f'{archive} - loss_out = {loss :.3f} - real_loss = {real_loss :.3f} - accuracy = {accuracy :.3f} - real_accuracy = {real_accuracy :.3f}')

        show(matrix,f'{real_loss}')

if __name__ == "__main__" : main()
