import cv2

# 使用opencv按一定间隔截取视频帧，并保存为图片

vc = cv2.VideoCapture('./video/test.mp4')  # 读取视频文件
c = 0
d = 0
print("------------")
if vc.isOpened():  # 判断是否正常打开
    print("yes")
    rval, frame = vc.read()
else:
    rval = False
    print("false")

timeF = 100  # 视频帧计数间隔频率

while rval:  # 循环读取视频帧
    rval, frame = vc.read()
    print(c,timeF, c%timeF)
    if (c % timeF == 0):# 每隔timeF帧进行存储操作
        print("write...")
        cv2.imwrite(f'./video_cut/1_{d}.jpg', frame)  # 存储为图像

        print("success!")
    c = c + 100000
    d = d + 1
cv2.waitKey(1)
vc.release()
print("==================================")
