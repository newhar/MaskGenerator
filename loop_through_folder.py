import cv2
import os
from mask import create_mask

FACE_LOC = "./face"

# face 폴더에 들어있는 마스크 착용하지 않은 이미지를 조회하며 마스크를 생성한다.
images = [os.path.join(FACE_LOC, f) for f in os.listdir(FACE_LOC) if os.path.isfile(os.path.join(FACE_LOC, f))]

for i in range(len(images)):
    print("the path of the image is", images[i])
    create_mask(images[i])



