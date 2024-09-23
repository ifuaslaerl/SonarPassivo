import torch
import scipy.io 
import os
from torch.utils.data import Dataset

class MatDataset(Dataset) :

    def __init__(self , main_path , transform=None) :
        self.main_path = main_path
        self.transform = transform
        self.data = []
        self.label = []

        self.classes = sorted(os.listdir(main_path))
        
        for idx , class_name in enumerate(self.classes) :
            class_path = os.path.join(main_path,class_name) 

            for archive in os.listdir(class_path) :
                mat_path = os.path.join(class_path,archive)
                mat_data = scipy.io.loadmat(mat_path)
                matriz = mat_data["ent_norm"]
                for i in range(len(matriz)):
                    self.data.append((mat_path,i))
                    self.label.append(idx)

    def __len__(self) : 
        return len(self.data)

    def __getitem__(self, index):
        
        mat_path , line_idx = self.data[index]
        label = self.label[index]

        mat_data = scipy.io.loadmat(mat_path)
        data = mat_data["ent_norm"][line_idx]

        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label,dtype=torch.long)

        if self.transform : data = self.transform(data)

        return data , label
