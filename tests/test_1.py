import math

import cv2
import mediapipe as mp

video = cv2.VideoCapture(0, cv2.CAP_DSHOW)
mpFaceMash = mp.solutions.face_mesh
faceMesh = mpFaceMash.FaceMesh()
mp_drawing = mp.solutions.drawing_utils

while True:
    check, img = video.read()
    img = cv2.resize(img, (1000, 720))
    imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = faceMesh.process(imgRGB)
    screenHeight, screenWidth, _ = img.shape
    #a

    if results:
        for face in results.multi_face_landmarks:
            #print(face)
            #mp_drawing.draw_landmarks(img, face,mpFaceMash.FACEMESH_FACE_OVAL)
            d11x, d11y = int((face.landmark[159].x) * screenWidth), int((face.landmark[159].y) * screenHeight)
            d12x, d12y = int((face.landmark[145].x) * screenWidth), int((face.landmark[145].y) * screenHeight)

            e11x, e11y = int((face.landmark[386].x) * screenWidth), int((face.landmark[386].y) * screenHeight)
            e12x, e12y = int((face.landmark[374].x) * screenWidth), int((face.landmark[374].y) * screenHeight)

            cv2.circle(img, (d11x, d11y), 1, (255, 0, 0), 2)
            cv2.circle(img, (d12x, d12y), 1, (255, 0, 0), 2)
            cv2.circle(img, (e11x, e11y), 1, (255, 0, 0), 2)
            cv2.circle(img, (e12x, e12y), 1, (255, 0, 0), 2)

            distanceRightEye = math.hypot(d11x - d12x, d11y - d12y)
            distanceLeftEye = math.hypot(e11x - e12x, e11y - e12y)

            if distanceRightEye <= 20 and distanceLeftEye <= 20:
                print("Olhos fechados")
                cv2.rectangle(img,(100, 30), (390,80), (0, 0, 255), -1)
                cv2.putText(img,"OLHOS FECHADOS", (105,65), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255), 3)
            else:
                print("Olhos abertos")
                cv2.rectangle(img, (100, 30), (370, 80), (0, 255,0), -1)
                cv2.putText(img, "OLHOS ABERTOS", (105, 65), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 3)

    cv2.imshow('IMG', img)
    cv2.waitKey(1)