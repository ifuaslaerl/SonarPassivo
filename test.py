#Bibliotecas

import torch
import torchvision.transforms as transforms
import os
import matplotlib.pyplot as plt
import MatDataset
import Networks

NE = 75
BS = 1 # tamanho dos conjuntos trabalhados
data = 'DadosSonar'
models_ = "Networks"
root = ".."

eps = 1e-6

def show(matriz,title):
    plt.title(title)
    plt.imshow(matriz , cmap='cividis')
    plt.colorbar()
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

    testset = MatDataset.MatDataset(f'{root}/{data}/test')

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
        
        real_loss , real_accuracy , matrix = Networks.test_loop(model,testloader)
        
        print(f'{archive} - loss_out = {loss :.3f} - real_loss = {real_loss :.3f} - accuracy = {accuracy :.3f} - real_accuracy = {real_accuracy :.3f}')

        if real_loss-0.09 >= eps : os.remove(os.path.join(models,archive))

        show(matrix,f'{real_loss}')

if __name__ == "__main__" : main()
