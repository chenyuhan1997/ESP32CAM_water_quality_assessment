import cv2
#url = 'http://192.168.43.244/mjpeg/1'
url = 'your ip stream'
cap = cv2.VideoCapture(url)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter('./video/test.mp4', fourcc, 20, (width, height))

while(cap.isOpened()):
    ret, frame = cap.read()
    if ret:
        out.write(frame)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) & 0xFF == ord('q'): #按键盘Q键退出
            break
    else:
        continue

cap.release()
out.release()
cv2.destroyAllWindows()

