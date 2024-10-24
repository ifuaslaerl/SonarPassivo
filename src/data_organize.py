""" Module providing functions to organize data. """
import os
import typing
from random import shuffle , seed
import numpy as np
from scipy.io import savemat , loadmat
from sklearn.model_selection import train_test_split

def modelate(path: str) -> None:
    """ Reestructure data in folder.

    Args:
        path (string): Path to folder.
    """

    # TODO Can be optmized is using os.makedirs() #pylint: disable=fixme
    for archive in os.listdir(path):
        new_path = os.path.join(path,archive)
        if new_path.endswith('.mat') :
            try :
                os.rename(new_path,f'{path}/{archive[0]}/{archive}')    
            except FileNotFoundError :
                os.mkdir(f'{path}/{archive[0]}')
                os.rename(new_path,f'{path}/{archive[0]}/{archive}')

def move(path1: str, path2: str) -> None:
    """ Move data from path1, to path2

    Args:
        path1 (string): Path where data is
        path2 (string): Path where data is going to
    """

    # TODO maybe can be optmized #pylint: disable=fixme
    for index , character in enumerate(path2):
        if character == '/' and not os.path.exists(path2[:index]) :
            os.mkdir(path2[:index])
    os.rename(path1,path2)

def sort_data(main_dir: str, new_dir: str, sections: typing.List[str], percent: typing.List) \
    -> None:
    """ Shuffle data in diferent sections.

    Args:
        main_dir (str): Path of main folder.
        new_dir (str): Path of result folder.
        sections (typing.List[str]): Sections names.
        percent (typing.List): List of percentage destinated to each folder.
    """

    # TODO review all script #pylint: disable=fixme

    for directorie in os.listdir(main_dir):
        new_path = os.path.join(main_dir,directorie)

        if len(directorie) != 1 :
            continue

        if 'A' > directorie[0] or directorie[0] > 'Z' :
            continue

        archives = os.listdir(new_path)
        shuffle(archives)

        size = len(archives)

        cont=0
        for i in range(size):

            if i+1 > round(percent[cont]*size) :
                cont += 1
            path2 = os.path.join(new_path,archives[i])
            new_path2 = f'{new_dir}/{sections[cont]}/{directorie}/{archives[i]}'
            move(path2,new_path2)

def counting(data: str) \
    -> typing.Tuple[typing.List[typing.List], typing.Dict[int, int], typing.Dict[int, int]]:
    """ Count the destribuition of data.

    Args:
        data (string): path of folder containing data.

    Returns:
        _type_: _description_
    """

    # TODO review all script #pylint: disable=fixme

    mapa = {}
    mapa2 = {}
    mapa3 = {}
    contador1=0
    contador2=0
    for i in os.listdir(data):
        path = os.path.join(data,i)
        folder = sorted(os.listdir(path))

        mapa[i] = contador1
        mapa2[contador1] = i
        contador1+=1

        for j in folder :
            path2 = os.path.join(path,j)
            folder = sorted(os.listdir(path2))

            for k in folder:
                if not k[:2] in mapa:
                    mapa[k[:2]] = contador2
                    mapa3[contador2] = k[:2]
                    contador2 += 1

    matriz = [ [ 0 for j in range(contador2) ] for i in range(contador1)]

    for conjunto in os.listdir(data):

        path = os.path.join(data,conjunto)
        folders = sorted(os.listdir(path))

        for classe in folders:
            path2 = os.path.join(path,classe)

            subfolders = sorted(os.listdir(path2))

            for arquivo in subfolders:
                matriz[mapa[conjunto]][mapa[arquivo[:2]]] += 1

    return matriz , mapa2 , mapa3

def show(new_dir: str) -> None:
    """ Show destribuition of data.

    Args:
        new_dir (str): Path of data.
    """

    # TODO review all script #pylint: disable=fixme

    matriz , mapa1 , mapa2 = counting(new_dir)

    print("#"*11,end="")
    for line in matriz:
        for jndex , value in enumerate(line):
            classe = mapa2[jndex]
            print(f'| {classe:2s} |',end="")
        break
    print()

    for index, line in enumerate(matriz):
        conjunto = mapa1[index]
        print(f'{conjunto:8s} - ',end="")
        for jndex, value in enumerate(line):
            print(f'| {value:2d} |',end="")
        print()

def advanced_sort_data(in_path: str,out_path: str, state=None) -> None:
    """ Sort data based in content of archives and write new data based in old one.

    Args:
        in_path (str): Path to input data folder.
        out_path (str): Path to output data folder.
        state (_type_, optional): Random state (seed). Defaults to None.
    """

    # TODO review all script #pylint: disable=fixme

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    folders = sorted(os.listdir(in_path))
    for classe in folders:
        class_path = os.path.join(in_path,classe)
        class_data = load_class_data(class_path)
        split_save(class_data,classe,out_path,state)

def load_class_data(class_path: str) -> np.array:
    """ Get data from path.

    Args:
        class_path (str): Path to data of determined label.

    Returns:
        np.array: All concatenated data.
    """

    class_data = []

    for arquivo in os.listdir(class_path):

        if not arquivo.endswith(".mat") :
            continue

        arch_path = os.path.join(class_path,arquivo)
        mat_data = loadmat(arch_path)
        matriz = mat_data["ent_norm"]
        class_data.append(matriz)

    return np.vstack(class_data)

def split_save(data: any, classe: str, out_path: str, state=None) -> None:
    """ Split and save data in Train, Test and Validate sets.

    Args:
        data (any): Data structure containing data.
        classe (str): Label to all data.
        out_path (str): Path to save all data.
        state (_type_, optional): Random state. Defaults to None.
    """

    if state:
        seed(state)

    np.random.shuffle(data)

    train_data, test_data = train_test_split(data, test_size=0.1)
    train_data , val_data = train_test_split(train_data , test_size=0.1)

    savemat(foldering(os.path.join(out_path, f'train/{classe}/{classe}_train.mat')) , \
        {'ent_norm': train_data})
    savemat(foldering(os.path.join(out_path, f'validate/{classe}/{classe}_val.mat')) , \
        {'ent_norm': val_data})
    savemat(foldering(os.path.join(out_path, f'test/{classe}/{classe}_test.mat')) , \
        {'ent_norm': test_data})

def foldering(string: str) -> str:
    """ Create all folders of determined path.

    Args:
        string (str): path to create folders.

    Returns:
        str: same path of input.
    """

    for index, character in enumerate(string):
        if character=="/":
            if not os.path.exists(string[:index]):
                os.mkdir(string[:index])

    return string

def find_name(path: str, extencion: str) -> str:
    """ Grants that the named file do not exist.

    Args:
        path (str): Original path to file.
        extencion (str): Type of file. ex: .csv

    Returns:
        str: New file name.
    """

    adition = 1
    while os.path.exists(path+str(adition)+extencion):
        adition+=1
    return path+str(adition)+extencion

if __name__ == "__main__":
    pass
