import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil
import math
import imutils

SEED_DATA_DIR = "C:/Base-de-Dados/BrainTumor"
num_of_images = {}

# Percorrer pasta de imagens/Criar pasta treino

for dir in os.listdir(SEED_DATA_DIR):
    num_of_images[dir] = len(os.listdir(os.path.join(SEED_DATA_DIR, dir )))
    
print(num_of_images)

# 70% treino, 15% validação e 15% teste

