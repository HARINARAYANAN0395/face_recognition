import cv2
camera=cv2.VideoCapture(0)
eye_detector=cv2.CascadeClassifier("eye_detector.xml")
face_detector=cv2.CascadeClassifier("facemodel.xml")
while True:
    success,frame=camera.read()
    grey_image=cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)
    eye_data=eye_detector.detectMultiScale(grey_image,minNeighbors=8,minSize=[30,30])
    face_data=face_detector.detectMultiScale(grey_image,minNeighbors=8,minSize=[30,30])
    # print(face_data)
    for x,y,w,h in eye_data :
        print(x,y,w,h)
        print(len(eye_data ))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
    for x,y,w,h in face_data :
        print(x,y,w,h)
        print(len(face_data ))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imshow("live video",frame)
    #print("haiiiiiiiiiiiiiiiiiiii")
    end=cv2.waitKey(1)
    if end==ord('q'):
            cv2.destroyAllWindows()
            camera.release()
            break