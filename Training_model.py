import numpy as np
import cv2
import os

# Address of the image.
Addres = 'C:\\Users\\matia\\Desktop\\FaceMask-Detection\\images' 
folderlist = os.listdir(Addres)
Labels = []
faces = []
cont = 0

for nd in folderlist:
    name = os.path.join(Addres, nd) # Reading the photos of the faces.
    if os.path.isdir(name):  # Verifying that name is a directory.
        for filename in os.listdir(name):
            Labels.append(cont)  # Assigning the labels.
            faces.append(cv2.imread(os.path.join(name, filename), 0))
        cont += 1

# Model and training.
F_recognition = cv2.face.LBPHFaceRecognizer_create()
F_recognition.train(faces, np.array(Labels))
F_recognition.write("TrainModel.xml")
print('The model has been succesfully created.')