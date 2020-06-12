import cv2
CAM_ID = 0
cap = cv2.VideoCapture(CAM_ID)

cv2.namedWindow('snow')


img_url="image/limkinam.jpg"


face_cascade=cv2.CascadeClassifier()
face_cascade.load('haarcascade.xml')


while(True):
    ret, frame = cap.read()
 
    grayframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    grayframe = cv2.equalizeHist(grayframe)

    faces = face_cascade.detectMultiScale(grayframe, 1.1, 3, 0, (30, 30))

    for (x,y,w,h) in faces:
        k=float(w)
        image = cv2.imread(img_url,1)
        r = k/image.shape[1]
        dim = (int(k), int(image.shape[0]*r))
        img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        imgROI = frame[y:y+h,x:x+w]
        img2 = img[0:h,0:w]

        img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)
        img1_fg = cv2.bitwise_and(img2, img2, mask =mask)
        dst = cv2.add(img1_fg, imgROI)
        frame[y:y+h,x:x+w]=dst
    cv2.imshow('snow',frame)
    if cv2.waitKey(1) >=0:
        break

cap.release()
cv2.destroyWindow('snow')