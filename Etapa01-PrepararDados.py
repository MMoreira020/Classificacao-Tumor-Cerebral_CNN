import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import shutil
import math
import imutils

SEED_DATA_DIR = "/home/rtx4060ti2/Documentos/BrainTumor"
num_of_images = {}

# Percorrer pasta de imagens/Criar pasta treino

for dir in os.listdir(SEED_DATA_DIR):
    num_of_images[dir] = len(os.listdir(os.path.join(SEED_DATA_DIR, dir )))
    
print(num_of_images)

# 70% treino, 15% validação e 15% teste

TRAIN_DIR = "/home/rtx4060ti2/Documentos/train"
VALIDATE_DIR = "/home/rtx4060ti2/Documentos/validate"
TEST_DIR = "/home/rtx4060ti2/Documentos/test"

# pasta treino
if not os.path.exists(TRAIN_DIR):
    os.mkdir(TRAIN_DIR)
    
    for dir in os.listdir(SEED_DATA_DIR):
        os.makedirs(TRAIN_DIR + "/" + dir)
        print(TRAIN_DIR + "/" + dir)
        
        for img in np.random.choice(a=os.listdir(os.path.join(SEED_DATA_DIR,dir)) , size= (math.floor(70/100* num_of_images[dir] )-5) , replace=False ):
            O = os.path.join(SEED_DATA_DIR, dir, img)
            print(O)
            D = os.path.join(TRAIN_DIR, dir)
            print(D)
            shutil.copy(O,D)
            os.remove(O)
else:
    print("Pasta treino existe!")
    
# pasta teste
if not os.path.exists(TEST_DIR):
    os.mkdir(TEST_DIR)
    
    for dir in os.listdir(SEED_DATA_DIR):
        os.makedirs(TEST_DIR + "/" + dir)
        print(TEST_DIR + "/" + dir)
        
        for img in np.random.choice(a=os.listdir(os.path.join(SEED_DATA_DIR,dir)) , size= (math.floor(15/100* num_of_images[dir] )-5) , replace=False ):
            O = os.path.join(SEED_DATA_DIR, dir, img)
            print(O)
            D = os.path.join(TEST_DIR, dir)
            print(D)
            shutil.copy(O,D)
            os.remove(O)
else:
    print("Pasta teste existe!")
    
# pasta validação
if not os.path.exists(VALIDATE_DIR):
    os.mkdir(VALIDATE_DIR)
    
    for dir in os.listdir(SEED_DATA_DIR):
        os.makedirs(VALIDATE_DIR + "/" + dir)
        print(VALIDATE_DIR + "/" + dir)
        
        for img in np.random.choice(a=os.listdir(os.path.join(SEED_DATA_DIR,dir)) , size= (math.floor(15/100* num_of_images[dir] )-5) , replace=False ):
            O = os.path.join(SEED_DATA_DIR, dir, img)
            print(O)
            D = os.path.join(VALIDATE_DIR, dir)
            print(D)
            shutil.copy(O,D)
            os.remove(O)
else:
    print("Pasta validacao existe!")