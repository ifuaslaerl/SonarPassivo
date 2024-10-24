""" Module providing analysis functions. """
import typing
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def to_csv(in_path: str, out_path: str):
    """ Receive an output file and make an .csv

    Args:
        in_path (str): input path of output file
        out_path (str): output path of .csv file
    """

    arch_read = open(in_path,"r" , encoding="UTF-8")
    arch_write = open(out_path,"w" , encoding="UTF-8")

    arch_write.write("Loss_in,Loss_out,Accuracy\n")

    for linha in arch_read:
        linha = linha.split()
        #print(linha[10][:-1]) # 4 7 10
        lista = [ linha[4] , linha[7] , linha[10][:-1] ]
        #print(lista)
        resposta = ""
        for data in lista : 
            resposta += data + ","
        #print(resposta)
        arch_write.write(resposta[:-1]+"\n")

def make_graph(in_path: str, out_path: str):
    """ Make a grafic of the information of .csv .

    Args:
        in_path (str): input_path of archive.cvs
        out_path (str): output_path of grafic
    """

    data = pd.read_csv(in_path , dtype=float)

    plt.plot(data["Loss_in"] , label="Loss_in")
    plt.plot(data["Loss_out"] , label="Loss_out")
    plt.plot(data["Accuracy"]/100 , label="Accuracy")

    plt.legend()
    plt.savefig(out_path)
    plt.show()
    plt.close()

def show_matrix(matrix: typing.List[typing.List] ,title : str):
    """ Make a grafic of confusion matrix.

    Args:
        matrix (typing.List[typing.List]): Confusion matrix.
        title (str): Title of grafic.
    """
    plt.title(title)
    sns.heatmap(matrix, annot=True, fmt="d", cmap="cividis", cbar=True, annot_kws={"size" : 16})
    plt.show()
    # TODO colocar um savefig aqui depois
