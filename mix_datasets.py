""" Module made to mix datasets. """
from src import mat_dataset

NE = 75
BS = 1 # tamanho dos conjuntos trabalhados

if __name__ == "__main__" :

    origin_set = mat_dataset.MatDataset("data/Datasets/DadosSonar/train")
    adversarial_set = mat_dataset.MatDataset("data/Datasets/DadosSonar/adversarial")

    origin_set.merge(adversarial_set,0.2)
    origin_set.save("data/Datasets/DadosSonar/adversarial_training")
