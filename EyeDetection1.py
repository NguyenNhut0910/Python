from math import hypot
import cv2
import dlib
import numpy as np

font = cv2.FONT_HERSHEY_SIMPLEX
cap = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def midpoint(p1,p2):
    return (p1.x + p2.x)//2, (p1.y + p2.y)//2

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame,(640,480))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        #print(face)
        landmarks = predictor(gray, face)
        '''
        for n in range(36, 48):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            cv2.circle(frame, (x, y), 3, (0, 255, 0), 1)
        '''
        left_point_r = (landmarks.part(36).x, landmarks.part(36).y)
        right_point_r = (landmarks.part(39).x, landmarks.part(39).y)
        center_top_r = midpoint(landmarks.part(37), landmarks.part(38))
        center_bottom_r = midpoint(landmarks.part(41), landmarks.part(40))
        #hor_line_r = cv2.line(frame, left_point_r, right_point_r, (0,255,0),1)
        #ver_line_r = cv2.line(frame, center_top_r, center_bottom_r, (0,255,0), 1)
        hor_line_r_length = hypot((left_point_r[0] - right_point_r[0]), (left_point_r[1] - right_point_r[1]))
        ver_line_r_lenght = hypot((center_top_r[0] - center_bottom_r[0]), (center_top_r[1]-center_bottom_r[1]))
        #print(ver_line_r_lenght, "-", hor_line_r_length)
        #print(hor_line_r_length/ver_line_r_lenght)
        ratio_r = hor_line_r_length/ver_line_r_lenght
        cv2.putText(frame, str(ratio_r), (10,450), font, 1, (0,0,255),1)
        if ratio_r > 5.2:
            cv2.putText(frame, "Sleepy...", (10,70), font, 2, (0,0,255),5)

            print(hor_line_r_length/ver_line_r_lenght)
        left_point_l = (landmarks.part(42).x, landmarks.part(42).y)
        right_point_l = (landmarks.part(45).x, landmarks.part(45).y)
        center_top_l = midpoint(landmarks.part(43), landmarks.part(44))
        center_bottom_l = midpoint(landmarks.part(47), landmarks.part(46))
        #hor_line_l = cv2.line(frame, left_point_l, right_point_l, (0,255,0),1)
        #ver_line_l = cv2.line(frame, center_top_l, center_bottom_l, (0,255,0), 1)

        
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1)&0xFF == 27: #key ESC
        break
 

cap.release()
cv2.destroyAllWindows()
