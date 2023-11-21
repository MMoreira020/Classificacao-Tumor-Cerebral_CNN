# verificação do modelo dentro de novos dados de teste
# execussão de predição em uma imagem 

import numpy as np
from keras.models import load_model
from keras.preprocessing import image
import keras
import cv2

TEST_DIR = "/home/rtx4060ti2/Documentos/test"

test_datagen = image.ImageDataGenerator( rescale= 1. /255)
test_data = test_datagen.flow_from_directory(directory=TEST_DIR, target_size=(224,224), batch_size=32, class_mode='binary')

# imprimir as classes

print("test_data.class_indices: ", test_data.class_indices)

# carregar modelo salvo

model = load_model('/home/rtx4060ti2/Documentos/MyBestModel.h5')

#print(model.summary() )

acc = model.evaluate(x=test_data)[1]

print(acc)



# load an image from the lest folder

#imagePath = "/home/rtx4060ti2/Documentos/test/Healthey/Not Cancer  (1525).jpg"
imagePath = "/home/rtx4060ti2/Documentos/test/Brain Tumor/Cancer (16).jpg"

img = image.load_img(imagePath, target_size=(224,224))
i = image.img_to_array(img)
i = i / 255
print(i.shape)


input_arr = np.array([i])
print(input_arr.shape)


predictions = model.predict(input_arr)[0][0]
print(predictions)

result = round(predictions)
if result == 0:
    text = 'Has a brain tumor'
else:
    text = 'Brain healthy'
    
print(text)

imgResult = cv2.imread(imagePath)
font = cv2.FONT_HERSHEY_COMPLEX
cv2.putText(imgResult, text, (0,20), font, 0.8, (255,0,0),2)
cv2.imshow('img', imgResult)
cv2.waitKey(0)
cv2.imwrite("/home/rtx4060ti2/Documentos/predictImage2.jpg", imgResult)