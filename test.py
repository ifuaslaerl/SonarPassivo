""" Module made to test Networks. """
import os
import torch
from src import networks, mat_dataset, analysis

NE = 75
BS = 1 # tamanho dos conjuntos trabalhados

if __name__ == "__main__" :
    DATA = "data/Datasets/DadosSonar"

    DEVICE = (
            "cuda"
            if torch.cuda.is_available()
            else "mps"
            if torch.backends.mps.is_available()
            else "cpu" )

    print(f'Using {DEVICE} device')

    testset = mat_dataset.MatDataset(f'{DATA}/test')

    testloader = torch.utils.data.DataLoader(testset,
                                        batch_size=BS,
                                        shuffle=True,
                                        num_workers=2)

    model = networks.SonarCNN(testset.classes)

<<<<<<< HEAD
    file_ = "0980.pth"

    check = torch.load(f"data/Networks/robust/{file_}")
=======

    check = torch.load("data/Networks/robust/0935.pth")
>>>>>>> 0f405ae50c0fec562440135a234378fe91c7a611
    model.load_state_dict(check['model_state_dict'])
    model.optimizer.load_state_dict(check['optimizer_state_dict'])
    loss = check['loss_out']
    accuracy = check['accuracy']
    model.eval()
    real_loss , real_accuracy , matrix = networks.adv_test_loop(model,testloader,0.01)
<<<<<<< HEAD
    print(f'{file_} - loss_out = {loss :.3f} - real_loss = {real_loss :.3f} - accuracy = {accuracy :.3f} - real_accuracy = {real_accuracy :.3f}')
=======
    print(f'0956.pth - loss_out = {loss :.3f} - real_loss = {real_loss :.3f} - accuracy = {accuracy :.3f} - real_accuracy = {real_accuracy :.3f}')
>>>>>>> 0f405ae50c0fec562440135a234378fe91c7a611
    analysis.show_matrix(matrix,f'{real_loss}')
