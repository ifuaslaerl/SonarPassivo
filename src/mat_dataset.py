""" Module providing MatDataset. """
from random import shuffle
import os
import typing
import numpy as np
import torch
import scipy.io
from torch.utils.data import Dataset

class MatDataset(Dataset) :
    """ Class representing a dataset. """

    def __init__(self, main_path: str, transform=None) :
        self.main_path = main_path
        self.transform = transform
        data = []
        label = []

        self.classes = sorted(os.listdir(main_path))

        for idx , class_name in enumerate(self.classes) :
            class_path = os.path.join(main_path,class_name) 

            for archive in os.listdir(class_path) :
                mat_path = os.path.join(class_path,archive)
                mat_data = scipy.io.loadmat(mat_path)
                matriz = mat_data["ent_norm"]
                for i in range(len(matriz)):
                    data.append((mat_path,i))
                    label.append(idx)

        self.data_label = list(zip(data,label))

    def __len__(self) -> int:
        return len(self.data_label)

    def __getitem__(self, index: int) -> typing.Tuple[torch.Tensor , torch.Tensor]:

        (mat_path , line_idx) , label = self.data_label[index]

        mat_data = scipy.io.loadmat(mat_path)
        data = mat_data["ent_norm"][line_idx]

        data = torch.tensor(data, dtype=torch.float32).unsqueeze(0)
        label = torch.tensor(label,dtype=torch.long)

        if self.transform:
            data = self.transform(data)

        return data , label

    def merge(self, dataset: 'MatDataset', percentage: float):
        """ Merge other dataset at this one.

        Args:
            dataset (MatDataset): Dataset to be merged.
        """

        new_dataset = []

        shuffle(dataset.data_label)
        for index, data in enumerate(dataset.data_label):
            new_dataset.append(data)
            #print(index, percentage*len(dataset.data_label))
            if index > percentage*len(dataset.data_label):
                break

        shuffle(self.data_label)
        for index, data in enumerate(self.data_label):
            new_dataset.append(data)
            #print(index, percentage*len(dataset.data_label))
            if index > (1-percentage)*len(self.data_label):
                break

        self.data_label = new_dataset

    def save(self, save_path=""):
        """_summary_

        Args:
            save_path (str, optional): Path to save data. Defaults to main path.
        """

        if save_path == "":
            save_path = self.main_path

        save_set(self.data_label, save_path, self.classes)

def save_set(dataset: typing.List, out_path: str, classes: typing.List[str]) -> None:
    """ Saves dataset.

    Args:
        dataset (typing.List): Core list of dataset to be saved.
        out_path (str): Path where dataset is going to be saved.
        classes (typing.List[str]): Classes of the datasets.
    """

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    organized_data = {}
    for data, label in dataset:
        #label = label.item()
        if label not in organized_data:
            organized_data[label] = []
        organized_data[label].append(data)

    for label, data in organized_data.items():
        data = np.concatenate(data)
        label_dir = os.path.join(out_path, classes[label])
        os.makedirs(label_dir, exist_ok=True)  # Criar diretório para o label
        mat_file_path = os.path.join(label_dir, f'{classes[label]}.mat')
        scipy.io.savemat(mat_file_path, {'ent_norm': np.array(data)})
