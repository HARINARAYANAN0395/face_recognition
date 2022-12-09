import cv2
camera=cv2.VideoCapture(0)
face_detector=cv2.CascadeClassifier("facemodel.xml")
while True:
    success,frame=camera.read()
    grey_image=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    face_data=face_detector.detectMultiScale(grey_image,minNeighbors=8,minSize=[30,30])
    # print(face_data)
    for x,y,w,h in face_data:
        print(x,y,w,h)
        print(len(face_data))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        cv2.putText(frame,"no of face detected "+str(len(face_data)),(100,200), cv2.FONT_HERSHEY_SIMPLEX , 1,(0,0,255),3)
    cv2.imshow("live video",frame)
    end=cv2.waitKey(1)
    if end==ord('q'):
            cv2.destroyAllWindows()
            camera.release()
            break