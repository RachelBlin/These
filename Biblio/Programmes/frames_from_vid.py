import cv2
vidcap = cv2.VideoCapture('/media/rblin/LaCie/seq0002 09-38-08.avi')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("/home/rblin/Documents/video_frames/frame%d.png" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1