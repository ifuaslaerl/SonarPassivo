import matplotlib.pyplot as plt

arquivo = open("Training_data/output3.txt","r")
arquivo2 = open("data.csv","w")

arquivo2.write("Loss_in,Loss_out,Accuracy\n")

for linha in arquivo:
    linha = linha.split()
    #print(linha[10][:-1]) # 4 7 10
    lista = [ linha[4] , linha[7] , linha[10][:-1] ]
    #print(lista)
    resposta = ""
    for data in lista : resposta += data + ","
    #print(resposta)
    arquivo2.write(resposta[:-1]+"\n")