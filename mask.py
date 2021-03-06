import os
import sys
import random
import argparse
import numpy as np
from PIL import Image, ImageFile

__version__ = '0.3.0'

# image load and save location

MASK_LOC = ["./images/default-mask.png", "./images/grey-mask.png"]
CORRECT_LOC = "./correct_mask/"
WRONG_LOC = "./wrong_mask/"

# const index for face_landmarks
CHIN_BOTTOM_IDX = 9
CHIN_LEFT_IDX = 3
CHIN_RIGHT_IDX = 15

NOSE_BRIDGE_IDX = 1

def create_mask(image_path, idx):
    pic_path = image_path
    mask_path = MASK_LOC[idx%2]
    show = False
    model = "hog"
    mask_on_face = False
    FaceMasker(pic_path, mask_path, show, model, mask_on_face).mask()


class FaceMasker:
    KEY_FACIAL_FEATURES = ('nose_bridge', 'chin')

    def __init__(self, face_path, mask_path, show=False, model='hog', mask_on_face=True):
        self.face_path = face_path
        self.mask_path = mask_path
        self.show = show
        self.model = model
        self.mask_on_face = mask_on_face
        self._face_img: ImageFile = None
        self._mask_img: ImageFile = None

    def mask(self):
        import face_recognition

        # face_location : returns an array of bound boxes of human face
        face_image_np = face_recognition.load_image_file(self.face_path)
        face_locations = face_recognition.face_locations(face_image_np, model=self.model)
        face_landmarks = face_recognition.face_landmarks(face_image_np, face_locations)
        self._face_img = Image.fromarray(face_image_np)
        self._mask_img = Image.open(self.mask_path)
        # self.mask_on_face = self.mask_on_face

        found_face = False
        for face_landmark in face_landmarks:
            # check whether facial features meet requirement
            skip = False
            for facial_feature in self.KEY_FACIAL_FEATURES:
                if facial_feature not in face_landmark:
                    skip = True
                    break
            if skip:
                continue

            # mask face
            found_face = True
            self._mask_face(face_landmark)

        if found_face:
            self._save()
        else:
            print('Found no face.')

    def _mask_face(self, face_landmark: dict):
        nose_bridge = face_landmark['nose_bridge']
        if self.mask_on_face:
            nose_point = nose_bridge[1] # 눈의 중간 지점 코 [28], nose_bridge[1]
        else:
            nose_point = nose_bridge[3]
        nose_v = np.array(nose_point)

        chin = face_landmark['chin']
        chin_len = len(chin)
        chin_bottom_point = chin[chin_len // 2] # 턱의 중간 [9]
        chin_bottom_v = np.array(chin_bottom_point)
        # print(chin_bottom_v)

        if not self.mask_on_face:
            chin_bottom_v[1] -= 30
        chin_left_point = chin[chin_len // 8] # 왼쪽턱 귀밑 [3]
        chin_right_point = chin[chin_len * 7 // 8] # 오른쪽 귀밑[15]

        # split mask and resize
        width = self._mask_img.width
        height = self._mask_img.height
        width_ratio = 1.2
        new_height = int(np.linalg.norm(nose_v - chin_bottom_v)) # 코뼈와 턱 아래 중심부와의 거리

        # left : to fit the left side of face, resize the mask
        mask_left_img = self._mask_img.crop((0, 0, width // 2, height)) #crop((left, top, right, bottom))
        mask_left_width = self.get_distance_from_point_to_line(chin_left_point, nose_point, chin_bottom_point)
        mask_left_width = int(mask_left_width * width_ratio)
        mask_left_img = mask_left_img.resize((mask_left_width, new_height))

        # right : to fit the right side of face, resize the mask
        mask_right_img = self._mask_img.crop((width // 2, 0, width, height))
        mask_right_width = self.get_distance_from_point_to_line(chin_right_point, nose_point, chin_bottom_point)
        mask_right_width = int(mask_right_width * width_ratio)
        mask_right_img = mask_right_img.resize((mask_right_width, new_height))

        # merge mask
        size = (mask_left_img.width + mask_right_img.width, new_height)
        mask_img = Image.new('RGBA', size)
        mask_img.paste(mask_left_img, (0, 0), mask_left_img)
        mask_img.paste(mask_right_img, (mask_left_img.width, 0), mask_right_img)

        # rotate mask : calculate between nose and chin
        angle = np.arctan2(chin_bottom_point[1] - nose_point[1], chin_bottom_point[0] - nose_point[0])
        rotated_mask_img = mask_img.rotate(angle, expand=True)

        # calculate mask location on the center
        center_x = (nose_point[0] + chin_bottom_point[0]) // 2
        center_y = (nose_point[1] + chin_bottom_point[1]) // 2

        offset = mask_img.width // 2 - mask_left_img.width
        radian = angle * np.pi / 180
        box_x = center_x + int(offset * np.cos(radian)) - rotated_mask_img.width // 2
        box_y = center_y + int(offset * np.sin(radian)) - rotated_mask_img.height // 2

        # print(box_x, box_y)
        # box_y += 60
        # add mask
        self._face_img.paste(mask_img, (box_x, box_y), mask_img)

    def _save(self):
        # save mask img on 'DEST_LOC'
        path_splits = os.path.splitext(self.face_path)
        # print(path_splits)
        mask_face_path = CORRECT_LOC + path_splits[0][7:9] + '_with_mask' + path_splits[1]
        if not self.mask_on_face:
            mask_face_path = WRONG_LOC + path_splits[0][7:9] + '_with_mask' + path_splits[1]

        self._face_img.save(mask_face_path)
        print(f'Save to {mask_face_path}')

    @staticmethod
    def get_distance_from_point_to_line(point, line_point1, line_point2):
        distance = np.abs((line_point2[1] - line_point1[1]) * point[0] +
                          (line_point1[0] - line_point2[0]) * point[1] +
                          (line_point2[0] - line_point1[0]) * line_point1[1] +
                          (line_point1[1] - line_point2[1]) * line_point1[0]) / \
                   np.sqrt((line_point2[1] - line_point1[1]) * (line_point2[1] - line_point1[1]) +
                           (line_point1[0] - line_point2[0]) * (line_point1[0] - line_point2[0]))
        return int(distance)


if __name__ == '__main__':
    create_mask(image_path)
