import cv2
import numpy as np
import face_recognition 


#STEP1 : loading the images and converting them into rgb(library understands only rgb)
imgsam= face_recognition.load_image_file('Students2/samantha.jpg')
imgsam= cv2.cvtColor(imgsam, cv2.COLOR_BGR2RGB)

imgshah= face_recognition.load_image_file('Students2/shahrukh.jpg')
imgshah= cv2.cvtColor(imgshah, cv2.COLOR_BGR2RGB)

imgaish= face_recognition.load_image_file('Students2/aishwariya.jpg')
imgaish= cv2.cvtColor(imgaish, cv2.COLOR_BGR2RGB)

imgarjun= face_recognition.load_image_file('Students2/arjun.jpg')
imgarjun= cv2.cvtColor(imgarjun, cv2.COLOR_BGR2RGB)


#STEP2 :Finding our face in image and finding encoding as well(pink box around face)
faceLocsam = face_recognition.face_locations(imgsam)[0]
encodesam = face_recognition.face_encodings(imgsam)[0]
cv2.rectangle(imgsam,(faceLocsam[3],faceLocsam[0]),(faceLocsam[1],faceLocsam[2]),(255,0,255),2) 
print(faceLocsam)

faceLocshah = face_recognition.face_locations(imgshah)[0]
encodeshah = face_recognition.face_encodings(imgshah)[0]
cv2.rectangle(imgshah,(faceLocshah[3],faceLocshah[0]),(faceLocshah[1],faceLocshah[2]),(255,0,255),2)

faceLocaish = face_recognition.face_locations(imgaish)[0]
encodeaish = face_recognition.face_encodings(imgaish)[0]
cv2.rectangle(imgaish,(faceLocaish[3],faceLocaish[0]),(faceLocaish[1],faceLocaish[2]),(255,0,255),2)

faceLocarjun = face_recognition.face_locations(imgarjun)[0]
encodearjun = face_recognition.face_encodings(imgarjun)[0]
cv2.rectangle(imgarjun,(faceLocarjun[3],faceLocarjun[0]),(faceLocarjun[1],faceLocarjun[2]),(255,0,255),2)


#STEP3 :Comparing 2 faces(comparing encodings i.e. 128 measurements) and finding distance between them
results1= face_recognition.compare_faces([encodesam],encodeshah)       
faceDis1= face_recognition.face_distance([encodesam],encodeshah)       
print(results1,faceDis1) 

results2= face_recognition.compare_faces([encodeaish],encodearjun)
faceDis2= face_recognition.face_distance([encodeaish],encodearjun)
print(results2,faceDis2)

cv2.putText(imgaish,f'{results2} {round(faceDis2[0],2)}',(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2) 


#STEP4: Displaying the finalised images on screen
cv2.imshow('Samantha Ruth', imgsam)       
cv2.imshow('ShahRukh Khan', imgshah)
cv2.imshow('Aishwariya Rai', imgaish)
cv2.imshow('Allu Arjun', imgarjun)
cv2.waitKey(0)