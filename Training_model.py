import numpy as np
import cv2
import os

# Dirección de la imagen
Addres = 'C:\\Users\\matia\\Desktop\\FaceMask-Detection\\images' 
folderlist = os.listdir(Addres)
Labels = []
faces = []
cont = 0

for nd in folderlist:
    name = os.path.join(Addres, nd) # Leer las fotos de los rostros
    if os.path.isdir(name):  # Asegurar que name es un directorio
        for filename in os.listdir(name):
            Labels.append(cont)  # Asignamos las etiquetas
            faces.append(cv2.imread(os.path.join(name, filename), 0))
        cont += 1

# Modelo y entrenamiento
F_recognition = cv2.face.LBPHFaceRecognizer_create()
F_recognition.train(faces, np.array(Labels))
F_recognition.write("TrainModel.xml")
print('The model has been succesfully created.')