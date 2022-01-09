#Author: Ömer Bera Dinç
#Defining Libraries
import cv2
import numpy as np
import pyzbar.pyzbar as pyzbar
#Capturing camera (for webcam:0, for usb cameras: 1+)
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
font = cv2.FONT_HERSHEY_SIMPLEX

#With Cascade-Trainer-GUI, I prepared a cascade.xml
qr_cascade = cv2.CascadeClassifier("qr-cascade/classifier/cascade.xml")

while True:
    ret, frame = cap.read()

    frame = cv2.flip(frame, 1)
    frame = cv2.resize(frame, (640, 480))
    qrs = qr_cascade.detectMultiScale(frame, 1.1, minNeighbors=2, minSize=(frame.shape[0]*10//64, frame.shape[1]*10//48))

    for (x, y, w, h) in qrs:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (20, 20, 255), 2)
        # cv2.rectangle(frame, (x-25, y-25), (x + w + 25, y + h + 25), (255, 20, 20), 2)
        roi = frame[y-25:y+h+25, x-25:x+w+25]

        try:
            qr_info = pyzbar.decode(roi)
            #gets qr data and defines as text
            text = qr_info[0].data
            text = text.decode("utf-8")
            #I am printing the data to the above of the qr code in the webcam recording
            cv2.putText(frame, text, (x-10, y-10), font, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

        except ZeroDivisionError:
            pass

        except IndexError:
            pass

    cv2.imshow("Video", frame)
    key = cv2.waitKey(1)
    #I assign a key to shutdown the webcam recording easily
    if key & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
