# 1. The Cv2 is the Library for the Open Vision and here 2 is the Version
import cv2



# 2. Face Classifier
# 2a. This below file is a pre trained model for detetcting faces when images pass it it will give in output 0 or 1. 
# 2b. It also support mutlifaces.
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
# 3. Smile Classifier  
# 3a. This below file is a pre trained model for detetcting smiles.
smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
# 4. Eyes Classifier  
# 4a. This below file is a pre trained model for detetcting Eyes.
eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')





# 5. Grab Webcam Feed this wat we can capture the video int VideoCapture
# 5b we can also put thr mp4 file.  
# 5c. else put 0 in args.
webcam = cv2.VideoCapture(0)




# To make a video  and Continous display of frame we put in Loop.
while True:

    # using .read() function we actually read the webcam and it return a tuple. 
    #The successful_frame_read actually gives a boolean output 1 and 0 
    # Frame it has the Cordinates.
    successful_frame_read, frame = webcam.read()


    # if there is Error Ocuured then ,Abort
    if not successful_frame_read:
        break

    # Change To Grayscale as the Actuall recogination of face is done in Black and white
    # As balck and white has one channel and rgb has three channel so it will less data to deal
    # But at the end we are going to work on Color Display only this only for detetcion fo face.
    frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)




    # Detects Face form the first as per the XML file instruction ..
    faces = face_detector.detectMultiScale(frame_grayscale)


    # This will create A rectangle around the faces form the cordinates it get form faces if it is more than on face it will create on all faces the green box.
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 4)
        # Args 1 is for Rectangle top left point  # Args 2 is for the size of Rectangle bottom right point  # Args 3 is for the Color of Rectangle  # Args 4 is for the Thickness of Rectangle



        # As eralier we have full image then we just want the subimage which is face so we are doing slicing on it As the smile will always inside the face only. .
        the_face = frame[y:y+h, x:x+w]

        # Here we are converting our face only nothing else not the whole frame to Black and white so it will be easy to detetct or smile.
        face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        # now it will only Detetct our smile  from the face_grayscale 
        smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)

        # now it will only Detetct our smile  from the face_grayscale 
        eyes = eye_detector.detectMultiScale(face_grayscale, scaleFactor=1.3, minNeighbors=10)




    # This will create A rectangle around the smile form the cordinates it get form smiles.
         for (x_, y_, w_, h_) in smiles:
            cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_),(50, 50, 200), 4)




    # This will create A rectangle around the eyes form the cordinates it get form eyes.
        for (x__, y__, w__, h__) in eyes:
            cv2.rectangle(the_face, (x__, y__), (x__ + w__, y__ + h__),(255, 255, 255), 4)


    # this will print the smiling word below the box fi the len is greater than 0
        if len(smiles) > 0:
            cv2.putText(frame, "Smiling", (x, y+h+40), fontScale=3,fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))



# imshow("First Args is Name of the winodw", "second will give the draw the picture on the winodw")
cv2.imshow('Smile Detector', frame)


# As the Above line show for split second so this will help us to wait untill the key is not pressed and we have to put one args otherwise it get current frame..
cv2.waitKey(1)


# This for Cleanup to realse the free resources.
webcam.release()
cv2.detoryAllWindows()
print("Code Compeleted")


