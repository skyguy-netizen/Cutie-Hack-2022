import numpy as np
import cv2 as cv

haar_cascade = cv.CascadeClassifier('haar_face.xml')

people = ['ben_afflek', 'elton_john', 'jerry_seinfeld', 'madonna', 'mindy_kaling']
features = np.load('features.npy', allow_pickle=1)
labels = np.load('labels.npy', allow_pickle=1)

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read('face_trained.yml')

cam = cv.VideoCapture(0)

while True:
    ret, frame = cam.read()
    cv.imshow("Cam", frame)

    k = cv.waitKey(1)
    if k%256 == 27:
        # ESC pressed
        print("Escape hit, closing...")
        break
    elif k%256 == 32:
        cv.imwrite('result.png', frame)


img = cv.imread('result.png')

gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

faces_rect = haar_cascade.detectMultiScale(gray, 1.1, 3)

for(x, y, w, h) in faces_rect:
    faces_roi = gray[y:y+h,x:x+h]

    label, confidence = face_recognizer.predict(faces_roi)
    cv.putText(img, people[label] + " " + str(confidence), (20,20), cv.FONT_HERSHEY_COMPLEX, 1.0, (0, 255, 0), thickness=2)
    cv.rectangle(img, (x,y), (x+w, y+h), (0, 255, 0), thickness=2)

cv.imshow("test", img)

cv.waitKey(0)

cam.release()

cv.destroyAllWindows()