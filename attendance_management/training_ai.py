import cv2
import numpy as np
from PIL import Image
import os
path = '/Users/harinarayanan/Documents/artificial intelligence/face_recognition/attendance_management/face_data'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier("facemodel.xml")
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]     
    faceSamples=[]
    ids = []
    for imagePath in imagePaths:
        image=cv2.imread(imagePath,0)
        cv2.imshow("test",image)
        cv2.waitKey(500)
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faceSamples.append(image)
        ids.append(id)
    return faceSamples,ids
print ("\n [INFO] Training faces. It will take a few seconds. Wait ...")
faces,ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))
# Save the model into trainer/trainer.yml
recognizer.write('trainer.xml') # recognizer.save() worked on Mac, but not on Pi
# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(len(np.unique(ids))))