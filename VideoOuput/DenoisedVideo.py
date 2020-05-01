import cv2
import numpy as np
import os
import sys
import glob




dataset_path = sys.argv[1]+'/*.jpg'




# for exporting video as .mp4
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

# for writing each output frame in asg04.mp4 file as output with 30 frame per second
img_array=[]
for filename in glob.glob(dataset_path):
    img = cv2.imread(filename)
    height, width, layers = img.shape
    size = (width,height)
    img_array.append(img)


    cv2.imshow('', img)

    # press 'q' to quit and save your output file
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
out = cv2.VideoWriter('Images2Video.mp4', fourcc, 10.0, size)
for i in img_array:
    out.write(i)
cv2.destroyAllWindows()
out.release()

