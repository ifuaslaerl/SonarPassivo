
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
from time import time

NF = 128
TK = 71
PO = 4
DR = 0.5
NN = 75

inf = 2e18

class Sonar_CNN(nn.Module):

    def __init__(self,classes):
        super(Sonar_CNN, self).__init__()

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
        
        x = F.relu(self.conv1d(x))
        x = self.maxpooling1d(x)    
        x = self.dropout(x)
        x = self.flatten(x)

        logits = self.dense(x)

        return logits

def train_loop(model,trainloader) :

    model.train()
    train_loss = 0
    for batch, (X, y) in enumerate(trainloader) :

        pred = model(X)
        loss = model.criterion(pred, y)
       
        train_loss += loss.item()

        loss.backward()
        model.optimizer.step()
        model.optimizer.zero_grad()

    return train_loss/len(trainloader)

def test_loop(model,dataloader) : 

    model.eval()
    classes = len(model.classes)
    validate_loss, correct = 0 , 0

    matriz = [[0 for i in range(classes)] for j in range(classes) ] 

    with torch.no_grad() :
        for X , y in dataloader:
                pred = model(X)
                validate_loss += model.criterion(pred, y).item()
                if pred.argmax(1) == y : correct += 1
                matriz[y-1][pred.argmax(1)-1] += 1 

    for i in range(len(matriz)) :   
        soma = sum(matriz[i])
        if soma : matriz[i] = [ round(100*matriz[i][j]/soma) for j in range(len(matriz[i])) ]
        else : matriz[i] = [0]*len(matriz[i])

    return validate_loss/len(dataloader) , correct/len(dataloader.dataset) , matriz

def fit(model,trainloader,validateloader,root,NE):

    start = time()
    minimum = inf
    for epoch in range(NE):
    
        loss_in = train_loop(model,trainloader)
        loss_out , accuracy , matriz = test_loop(model,validateloader)

        print(f'Epoch [{epoch+1}/{NE}] - Loss_in: {loss_in :.3f} - Loss_out: {loss_out :.3f} - Accuracy: {(100*accuracy):>0.1f}% - in {int(time()-start) :03d} seconds')
        
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


def FGSM(sample, eps, data_grad):

    sign_data_grad = data_grad.sign()

    adv_sample = sample + eps*sign_data_grad
    
    return adv_sample


def adv_test_loop(model,dataloader,eps) : 

    torch.multiprocessing.set_sharing_strategy('file_system')

    model.eval()
    classes = len(model.classes)
    adv_loss, correct = 0 , 0

    matriz = [[0 for i in range(classes)] for j in range(classes) ] 

    for data , label in dataloader:
        
        data.requires_grad = True

        output = model(data)
        init_pred = output.max(1, keepdim=True)[1]

        #if init_pred.item() != target.item(): continue

        loss = model.criterion(output, label)

        model.zero_grad()
        loss.backward()
        data_grad = data.grad.data

        adv_sample = FGSM(data, eps, data_grad)

        adv_output = model(adv_sample)
        final_pred = adv_output.max(1, keepdim=True)[1]
        adv_loss += model.criterion(adv_output, label)

        if final_pred.item() == label.item(): correct += 1
        matriz[label.item()-1][final_pred.item()-1] += 1
            
    for i in range(len(matriz)) :   
        soma = sum(matriz[i])
        if soma : matriz[i] = [ round(100*matriz[i][j]/soma) for j in range(len(matriz[i])) ]
        else : matriz[i] = [0]*len(matriz[i])

    return adv_loss/len(dataloader) , correct/len(dataloader.dataset) , matriz
