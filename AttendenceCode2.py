import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime


#STEP1: Create a list that will get the images from our folder automatically
path = 'Students2'              
images = []                                                   
classNames = []                
myList = os.listdir(path)      
print(myList)                 

for cls in myList:                                 
    curImg = cv2.imread(f'{path}/{cls}')          
    images.append(curImg)                          
    classNames.append(os.path.splitext(cls)[0])    
print(classNames) 


#STEP2: Then, function will generate the encoding for imported images automatically
def findEncodings(images): 
    encodeList = []                                        
    for img in images:                                     
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)                                                          
        encode = face_recognition.face_encodings(img)[0]   
        encodeList.append(encode)                          
    return encodeList    


#STEP4: Mark our Attendence and fill the csv file with name,time
def markAttendence(name):
    with open('AttendenceRecord2.csv', 'r+') as f :
        myDataList = f.readlines()
        print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtstring = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtstring}')


encodeListKnown = findEncodings(images)             
print(len(encodeListKnown))                         
print("Encoding complete!")


#STEP3: To find the matches between our encoded image and other(which we dont have)
#so, to get matching image ,it will try to find it in our webcam

#--intialising the webcam ([0]=id of webcam) 
cap = cv2.VideoCapture(0)   
while True:                                                      
    success, img = cap.read()                                    
    img_small = cv2.resize(img,(0,0),None,0.25,0.25)                                                                     
    img_small = cv2.cvtColor(img_small,cv2.COLOR_BGR2RGB)
#--find the encoding of the webcam
    facesCurFrame = face_recognition.face_locations(img_small)   
    encodesCurFrame = face_recognition.face_encodings(img_small,facesCurFrame) 

#--next, finding the matches
   #will iterate through all the faces that we have found in our current frame 
   #then we will compare all these faces with all the encodings that we found before
    for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):          
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace) 
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace) 
        #print(faceDis)
        matchIndex = np.argmin(faceDis)
        #print(matchIndex) 

#--now we know the person,then, display bounding box around them and write their name
        if matches[matchIndex]:                       
            name = classNames[matchIndex].upper()    
            #print(name)                              
            y1,x2,y2,x1 = faceLoc                                  
            y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4               
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)              
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED) 
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)   
                                                                        
            markAttendence(name)    

    cv2.imshow('STUDENT WEBCAM',img)       
    cv2.waitKey(1)    
