""" Module providing neuron network functions. """
import os
import typing
from time import time
import scipy.io
from torch import optim
from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

NF = 128
TK = 71
PO = 4
DR = 0.5
NN = 75
INF = 2e18

class SonarCNN(nn.Module):
    """ Class representing Neuron Network to categorize Ships and submarines. """

    def __init__(self,classes):
        super().__init__()

        self.classes = classes
        self.conv1d = nn.Conv1d(1 , NF , TK)
        self.maxpooling1d = nn.MaxPool1d(PO)
        self.dropout = nn.Dropout(DR)
        self.flatten = nn.Flatten()

        self.dense = nn.Sequential(
                    nn.Linear(121*128,NN),
                    nn.ReLU(),
                    nn.Linear(NN,len(self.classes))
                )

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        """ Foward information to next node.

        Args:
            x (Tensor): information in node.

        Returns:
            Tensor: information in fowarded node.
        """

        x = F.relu(self.conv1d(x))
        x = self.maxpooling1d(x)
        x = self.dropout(x)
        x = self.flatten(x)
        logits = self.dense(x)

        return logits

def train_loop(model : SonarCNN, trainloader: torch.utils.data.dataloader.DataLoader) -> float:
    """ Train loop of a neural network.

    Args:
        model (SonarCNN): Neural Network.
        trainloader (torch.utils.data.dataloader.DataLoader): Loader with trainset.

    Returns:
        float: Avarege loss during training.
    """

    model.train()
    train_loss = 0
    for batch, (x, y) in enumerate(trainloader) :

        pred = model(x)
        loss = model.criterion(pred, y)

        train_loss += loss.item()

        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()

    return train_loss/len(trainloader)

def test_loop(model: SonarCNN,dataloader: torch.utils.data.dataloader.DataLoader) \
    -> typing.Tuple[float, float, typing.List[typing.List]]:
    """ Test loop to evaluate Neuron Network.

    Args:
        model (SonarCNN): Neuron Network.
        dataloader (torch.utils.data.dataloader.DataLoader): Loader with testing data.

    Returns:
        typing.Tuple[float, float, typing.List[typing.List]]: \
            avarage loss, accuracy and confusion matrix.
    """

    model.eval()
    validate_loss, correct = 0 , 0

    matriz = [[0 for i in model.classes] for j in model.classes]

    with torch.no_grad() :
        for x , y in dataloader:
            pred = model(x)
            validate_loss += model.criterion(pred, y).item()
            if pred.argmax(1) == y :
                correct += 1
            matriz[y-1][pred.argmax(1)-1] += 1

    matrix = []
    for line in matriz:
        soma = sum(line)
        if soma:
            line = [round(100*value/soma) for value in line ]
        else:
            line = [0]*len(line)
        matrix.append(line)

    return validate_loss/len(dataloader) , correct/len(dataloader.dataset) , matrix

def fit(model: SonarCNN, trainloader: torch.utils.data.dataloader.DataLoader,\
    validateloader: torch.utils.data.dataloader.DataLoader,\
    root: str, numer_of_epochs: int) -> None:
    """ Fit Neural Network with epochs of train loops.

    Args:
        model (SonarCNN): Neuron Network.
        trainloader (torch.utils.data.dataloader.DataLoader): Loader with train dataset.
        validateloader (torch.utils.data.dataloader.DataLoader): Loader with validate dataset.
        path (str): Path to save Networks.
        numer_of_epochs (int): Number of epochs.
    """

    start = time()
    minimum = INF
    for epoch in range(numer_of_epochs):

        loss_in = train_loop(model,trainloader)
        loss_out , accuracy , matriz = test_loop(model,validateloader)

        print(f'Epoch [{epoch+1}/{numer_of_epochs}] - Loss_in: {loss_in :.3f} - Loss_out: {loss_out :.3f} - Accuracy: {(100*accuracy):>0.1f}% - in {int(time()-start) :03d} seconds')

        if minimum > loss_out :
            minimum = loss_out

            file_name = f'Networks/{int(1000*accuracy) :04d}.pth'

            torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.optimizer.state_dict(),
            'epoch': epoch,    
            'loss_in' : loss_in,
            'loss_out': loss_out,   
            'accuracy' : accuracy,
            'confusion' : matriz
            }, os.path.join(root,file_name))

def fgsm(sample: any, eps: float, data_grad: torch.Tensor) -> any:
    """ Generate adversarial data from sample and loss gradient. 

    Args:
        sample (any): Original sample.
        eps (float): Intensity of attack.
        data_grad (torch.Tensor): Loss gradient.

    Returns:
        any: Adversarial Sample.
    """

    sign_data_grad = data_grad.sign()
    adv_sample = sample + eps*sign_data_grad

    return adv_sample

def adv_test_loop(model: SonarCNN,dataloader: torch.utils.data.dataloader.DataLoader,\
    eps: float) -> typing.Tuple[float, float, typing.List[typing.List]]:
    """ Test a Neural Network throug adversarial attacks of determined intensity.

    Args:
        model (SonarCNN): Neural Network.
        dataloader (torch.utils.data.dataloader.DataLoader): Loader with dataset to be tested.
        eps (float): Intensity of attack.

    Returns:
        typing.Tuple[float, float, typing.List[typing.List]]: \
            avarage loss, accuracy and confusion matrix.
    """

    torch.multiprocessing.set_sharing_strategy('file_system')

    model.eval()
    classes = len(model.classes)
    adv_loss, correct = 0 , 0

    matrix = [[0 for i in classes] for j in classes]

    for data , label in dataloader:

        data.requires_grad = True

        output = model(data)

        #init_pred = output.max(1, keepdim=True)[1]
        #if init_pred.item() != target.item(): continue

        loss = model.criterion(output, label)

        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        adv_sample = fgsm(data, eps, data_grad)

        adv_output = model(adv_sample)
        final_pred = adv_output.max(1, keepdim=True)[1]
        adv_loss += model.criterion(adv_output, label)

        if final_pred.item() == label.item():
            correct += 1
        matrix[label.item()-1][final_pred.item()-1] += 1

    for line in matrix:
        soma = sum(line)
        if soma:
            line = [round(100*value/soma) for value in line]
        else:
            line = [0]*len(line)

    return adv_loss/len(dataloader) , correct/len(dataloader.dataset) , matrix

def adv_data_gen(model : SonarCNN, trainloader: torch.utils.data.dataloader.DataLoader,\
    eps: float) -> typing.List :
    """ Generate adversarial samples from determined loader and network.

    Args:
        model (SonarCNN): Neural Network
        trainloader (torch.utils.data.dataloader.DataLoader): Loader with dataset.
        eps (float): Intensity of attacks

    Returns:
        typing.List: List containing adversarial data and label
    """

    print("Adversarial data is beeing generated.")

    torch.multiprocessing.set_sharing_strategy('file_system')

    model.eval()

    adversarial_samples = []
    labels_samples = []

    start = time()

    for index, (data, label) in enumerate(trainloader):

        data.requires_grad = True
        pred = model(data)
        init_pred = pred.max(1, keepdim=True)[1]

        if init_pred != label:
            continue

        loss = model.criterion(pred, label)
        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        adv_sample = fgsm(data, eps, data_grad)

        adversarial_samples.append(adv_sample)
        labels_samples.append(label)
        print(f"{100*index/len(trainloader) : .1f}% of adversarial dataset generated.")

    adversarial_dataset = list(zip(adversarial_samples, labels_samples))

    print(f"Adverdarial dataset generated in {round(time()-start)} seconds.")
    return adversarial_dataset

def save_set(dataset: typing.List, out_path: str, classes: typing.List[str]) -> None:

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    organized_data = {}
    for data, label in dataset:
        label = label.item()
        if label not in organized_data:
            organized_data[label] = []
        organized_data[label].append(data.detach().numpy())

    for label, data in organized_data.items():
        data = np.concatenate(data)
        label_dir = os.path.join(out_path, classes[label])
        os.makedirs(label_dir, exist_ok=True)  # Criar diretório para o label
        mat_file_path = os.path.join(label_dir, f'{classes[label]}.mat')
        scipy.io.savemat(mat_file_path, {'ent_norm': np.array(data)})
