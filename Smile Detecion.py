def smiledtetctor():
    import cv2
    face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    smile_detector = cv2.CascadeClassifier('haarcascade_smile.xml')
    eye_detector = cv2.CascadeClassifier('haarcascade_eye.xml')

    webcam = cv2.VideoCapture(0)

    while True:

        successful_frame_read, frame = webcam.read()

        if not successful_frame_read:
            break


        frame_grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(frame_grayscale)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (100, 200, 50), 4)

            the_face = frame[y:y+h, x:x+w]
            face_grayscale = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

            smiles = smile_detector.detectMultiScale(face_grayscale, scaleFactor=1.7, minNeighbors=20)
            eyes = eye_detector.detectMultiScale(face_grayscale, scaleFactor=1.3, minNeighbors=10)

            for (x_, y_, w_, h_) in smiles:
                cv2.rectangle(the_face, (x_, y_), (x_ + w_, y_ + h_),
                              (50, 50, 200), 4)

            for (x__, y__, w__, h__) in eyes:
                cv2.rectangle(the_face, (x__, y__), (x__ + w__, y__ + h__),
                              (255, 255, 255), 4)

            if len(smiles) > 0:
                cv2.putText(frame, "Smiling", (x, y+h+40), fontScale=3,
                            fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))


        cv2.imshow('Smile Detector', frame)
        cv2.waitKey(1)

    webcam.release()
    cv2.detoryAllWindows()
    print("Code Compeleted")


if __name__ == "__main__":
    smiledtetctor()
