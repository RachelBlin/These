import cv2
vidcap = cv2.VideoCapture('/media/rblin/LaCie/Aquisitions_goPro_Polar/07_05_2019_16h30/RGB/videos/GP040398.MP4')
success,image = vidcap.read()
count = 0
while success:
  cv2.imwrite("/media/rblin/LaCie/Aquisitions_goPro_Polar/07_05_2019_16h30/RGB/frames/vid5/frame%d.png" % count, image)     # save frame as JPEG file
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  count += 1