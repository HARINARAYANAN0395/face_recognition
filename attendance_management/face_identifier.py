import cv2
import numpy as np
import os
from gtts import gTTS  


recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer.xml')
cascadePath = "/Users/harinarayanan/Documents/artificial intelligence/face_recognition/attendance_management/facemodel.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX
#iniciate id counter
id = 0
# names related to ids: example ==> Marcelo: id=1,  etc
names = {"unknown": " UNKNOWN","123":"harinarayanan","124":"apsitha",}
# Initialize and start realtime video capture
# cam = cv2.VideoCapture(0)
cam = cv2.VideoCapture(0)

f = open("newfile.txt", "a")
while True:
    ret, img =cam.read()
    #img = cv2.flip(img, -1) # Flip vertically
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale( 
        gray,
        scaleFactor = 1.2,
        minNeighbors = 5,
       )
    for(x,y,w,h) in faces:
        cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
        id, loss = recognizer.predict(gray[y:y+h,x:x+w])
        # Check if confidence is less them 100 ==> "0" is perfect match 
        print(id,loss)
        if (loss <40):
            # loss = " {0}%".format(round(100 - confidence))
            cv2.putText(img, names[str(id)], (x+5,y-5), font, 1, (255,0,255), 2)
            cv2.putText(img, str(loss), (x+5,y+h-5), font, 1, (255,255,0), 1)  
            print(id)
            # f.write(id)
            f.write("\n")
        else:
            cv2.putText(img, names["unknown"], (x+5,y-5), font, 1, (255,0,255), 2)
            cv2.putText(img, str(loss), (x+5,y+h-5), font, 1, (255,255,0), 1)  
        
        
    
    cv2.imshow('camera',img) 
    k = cv2.waitKey(1) & 0xff # Press 'ESC' for exiting video
    if k == 27:
        f.close()
        break

# Do a bit of cleanup
print("\n [INFO] Exiting Program and cleanup stuff")
cam.release()
cv2.destroyAllWindows()