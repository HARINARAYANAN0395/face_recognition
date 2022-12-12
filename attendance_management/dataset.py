import cv2
import os
camera=cv2.VideoCapture(0)
face_detector=cv2.CascadeClassifier("/Users/harinarayanan/Documents/artificial intelligence/face_recognition/attendance_management/facemodel.xml")
face_id=input("/n enter user id and <return>==>")
count=0
while (True):
    ret,img=camera.read()
    success,frame=camera.read()
    grey_image=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    face_data=face_detector.detectMultiScale(grey_image,minNeighbors=8,minSize=[30,30])
    for x,y,w,h in face_data:
        print(x,y,w,h)
        print(len(face_data))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        count+=1
        cv2.imwrite("/Users/harinarayanan/Documents/artificial intelligence/face_recognition/attendance_management/face_data/user." + str(face_id) + '.' + str(count) + ".jpg", frame[y:y+h,x:x+w])
        cv2.imshow('image', img)
    cv2.imshow("live video",frame)
    end=cv2.waitKey(10)
    if end==ord('q'):
        cv2.destroyAllWindows()
        camera.release()
        break
    elif count>=30:
        break
    print("\n [INFO] Exiting Program and cleanup stuff")
camera.release()
cv2.destroyAllWindows()