import cv2
vidcap = cv2.VideoCapture('/home/rblin/Documents/Databases/11_05_2019/videos/GP040399.MP4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("/home/rblin/Documents/Databases/11_05_2019/frames/vid5/frame%d.png" % count, image)     # save frame as png file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1