import cv2 as cv
import os
from keras.models import load_model
import numpy as np

model = load_model('./model2-010.model')
predResult = {0: 'No Mask', 1: 'Mask'}
frameColor = {0: (255,0,0), 1: (0,255,0)}

def FaceTracker():
    faceCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    cam = cv.VideoCapture(0)

    cv.namedWindow("test")

    img_counter = 0

    while True:
        ret, frame = cam.read()
        ogImg = frame
        if not ret:
            print("failed to grab frame")
            break
        frame = cv.flip(frame,1,1)
        frameGray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(frameGray, 1.3, 5) 
        for (x, y, w, h) in faces:
            frameImg = frame[y:y+h,x:x+w]
            frame_size = cv.resize(frameImg, (150,150))
            normal = frame_size/255.0
            reshaped = np.reshape(normal,(1,150,150,3))
            reshaped = np.vstack([reshaped]) 
            result=model.predict(reshaped)
            label= np.argmax(result,axis=1)[0]
            cv.rectangle(frame, (x, y), (x + w, y + h), frameColor[label], 2)
            cv.rectangle(frame,(x,y-40),(x+w,y),frameColor[label],-1)
            cv.putText(frame, predResult[label], (x,y-10), cv.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)
        cv.imshow("test", frame)

        k = cv.waitKey(1)
        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            img_name = "opencv_frame_{}.png".format(img_counter)
            cv.imwrite(img_name, ogImg)
            print("{} written!".format(img_name))
            img_counter += 1

    cam.release()

    new_img_counter = 0
    while new_img_counter <= img_counter:
        img = cv.imread("opencv_frame_{}.png".format(new_img_counter))
        canny = cv.Canny(img, 100, 175) 
        # gray = cv.cvtColor(canny, cv.COLOR_BGR2GRAY)
        # edges  = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 9, 8) 
        # blurred = cv.medianBlur(result, 3)
        # cartoon = cv.bitwise_and(blurred, blurred, mask=edges)     
        # canny = cv.cvtColor(canny,cv.COLOR_BGR2GRAY)
        # canny = cv.medianBlur(canny,3)
        # canny = cv.adaptiveThreshold(canny,255,cv.ADAPTIVE_THRESH_MEAN_C,cv.THRESH_BINARY,5,5)
        # cartoon = cv.bitwise_and(canny,canny,mask=edges)
        img_name = "opencv_frame_{}.png".format(new_img_counter)
        cv.imwrite(img_name, canny)
        new_img_counter += 1

    # editImg = cv.canny(img_name,125,175)
    # cv.imshow("Edited",editImg)
        command = "open " + img_name
        print(command)
        os.system(command)
    # cv.waitKey(0)
    cv.destroyAllWindows()

FaceTracker()


# # window.destroy()

# import cv2
# import numpy as np
# from keras.models import load_model
# model=load_model("./model2-001.model")

# results={0:'without mask',1:'mask'}
# GR_dict={0:(0,0,255),1:(0,255,0)}

# rect_size = 4
# cap = cv2.VideoCapture(0) 


# haarcascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# while True:
#     (rval, im) = cap.read()
#     im=cv2.flip(im,1,1) 

    
#     rerect_size = cv2.resize(im, (im.shape[1] // rect_size, im.shape[0] // rect_size))
#     faces = haarcascade.detectMultiScale(rerect_size)
#     for f in faces:
#         (x, y, w, h) = [v * rect_size for v in f] 
        
#         face_img = im[y:y+h, x:x+w]
#         rerect_sized=cv2.resize(face_img,(150,150))
#         normalized=rerect_sized/255.0
#         reshaped=np.reshape(normalized,(1,150,150,3))
#         reshaped = np.vstack([reshaped])
#         result=model.predict(reshaped)

        
#         label=np.argmax(result,axis=1)[0]
      
#         cv2.rectangle(im,(x,y),(x+w,y+h),GR_dict[label],2)
#         cv2.rectangle(im,(x,y-40),(x+w,y),GR_dict[label],-1)
#         cv2.putText(im, results[label], (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

#     cv2.imshow('LIVE',   im)
#     key = cv2.waitKey(10)
    
#     if key == 27: 
#         break

# cap.release()

# cv2.destroyAllWindows()