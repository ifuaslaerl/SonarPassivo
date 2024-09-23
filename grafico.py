import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("data.csv" , dtype=float)

plt.plot(data["Loss_in"] , label="Loss_in")
plt.plot(data["Loss_out"] , label="Loss_out")
#plt.plot(data["Accuracy"]/100 , label="Accuracy")

plt.legend()
plt.savefig("Graficos/grafico3.png")