import os
import numpy as np
from scipy.io import savemat , loadmat
from sklearn.model_selection import train_test_split
from random import shuffle , seed

def modelate(path):
    for archive in os.listdir(path):
        new_path = os.path.join(path,archive)
        if new_path.endswith('.mat') :
            try :
                os.rename(new_path,f'{path}/{archive[0]}/{archive}')    
            except FileNotFoundError :
                os.mkdir(f'{path}/{archive[0]}')
                os.rename(new_path,f'{path}/{archive[0]}/{archive}')
    
def move(path1 , path2):
        
    for i in range(len(path2)):
        if path2[i] == '/' and not os.path.exists(path2[:i]) :
            os.mkdir(path2[:i])
    os.rename(path1,path2)
        
def sort_data(main_dir,new_dir,sections):
    
    for directorie in os.listdir(main_dir):
        new_path = os.path.join(main_dir,directorie)
        
        if len(directorie) != 1 : continue
        if 'A' > directorie[0] or directorie[0] > 'Z' : continue   
        
        archives = os.listdir(new_path)
        shuffle(archives)
        
        size = len(archives)
        
        cont=0
        for i in range(size):
        
            if i+1 > round(percent[cont]*size) : cont += 1
            path2 = os.path.join(new_path,archives[i])
            new_path2 = f'{new_dir}/{sections[cont]}/{directorie}/{archives[i]}'
            move(path2,new_path2)     

def counting(data):

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

def show(new_dir):

    matriz , mapa1 , mapa2 = counting(new_dir)
    
    print("#"*11,end="")
    for i in range(len(matriz)):
        for j in range(len(matriz[i])):
            classe = mapa2[j]
            print(f'| {classe:2s} |',end="")
        break
    print()

    for i in range(len(matriz)):
        conjunto = mapa1[i]
        print(f'{conjunto:8s} - ',end="")
        for j in range(len(matriz[i])):
            classe = matriz[i][j]
            print(f'| {classe:2d} |',end="")
        print() 

def advanced_sort_data(in_path,out_path,state=None):

    if not os.path.exists(out_path):
        os.makedirs(out_path)

    folders = sorted(os.listdir(in_path))
    for classe in folders:
        class_path = os.path.join(in_path,classe) 
        
        class_data = load_class_data(class_path)

        split_save(class_data,classe,out_path,state)

def load_class_data(class_path):
    class_data = []
    
    for arquivo in os.listdir(class_path):
        if not arquivo.endswith(".mat") : continue 
        arch_path = os.path.join(class_path,arquivo)
        mat_data = loadmat(arch_path)
        matriz = mat_data["ent_norm"]
        class_data.append(matriz)

    return np.vstack(class_data)

def split_save(data,classe,out_path,state=None):
    if state: seed(state)

    np.random.shuffle(data)
    
    train_data, test_data = train_test_split(data, test_size=0.1)
    train_data , val_data = train_test_split(train_data , test_size=0.1)
    
    savemat(foldering(os.path.join(out_path, f'train/{classe}/{classe}_train.mat')) , {'ent_norm': train_data})
    savemat(foldering(os.path.join(out_path, f'validate/{classe}/{classe}_val.mat')) , {'ent_norm': val_data})
    savemat(foldering(os.path.join(out_path, f'test/{classe}/{classe}_test.mat')) , {'ent_norm': test_data})

def foldering(string):
    
    for i in range(len(string)):
        if string[i]=="/":
            if not os.path.exists(string[:i]):
                os.mkdir(string[:i])
    
    return string

if __name__ == "__main__":
    
    #modelate("Dados_SONAR")
    #advanced_sort_data("Dados_SONAR","DadosSonar",state=15262)
    #show(new_dir)    
    
    pass
