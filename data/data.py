import cv2
import torch
import random

from fontTools.ttLib.woff2 import bboxFormat
from torch.utils.data import Dataset
import albumentations as A
IMAGE_SIZE = 512
def make_transforms():
    """Preserve the full image; apply random deformation only during training."""

    def resize_and_pad():
        a = A.Compose([
            A.LongestMaxSize(
                max_size=IMAGE_SIZE,
                interpolation=cv2.INTER_LINEAR)]
            +
            [A.PadIfNeeded(
                min_height=IMAGE_SIZE,
                min_width=IMAGE_SIZE,
                border_mode=cv2.BORDER_CONSTANT,
                fill=0,
            )],
            bbox_params=A.BboxParams(
            format='pascal_voc',  # 1. 指定传入的格式
            label_fields=['category_ids'],  # 2. 告诉库对应的类别标签变量名
            min_visibility=0.3))
        return a

    train_transform = A.Compose(
        [
            A.ElasticTransform(
        alpha=1,
        sigma=50,
        interpolation=cv2.INTER_LINEAR,
        border_mode=cv2.BORDER_CONSTANT,
        fill=0,
        p = 0.8
    )]
        + [
            A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit= 0.1,
                contrast_limit= 0.1, p=1),
            A.GaussianBlur(
                blur_limit=(3,3), p=1),
                #blur_limit = kernel size
            A.RandomGamma(gamma_limit=(80, 120), p=1.0)
                #change the contrast/brightness[0.8,1.2]
        ], p=0.8)]

        + [
            A.Flip(p=0.5),
            A.Affine(
                rotate = (-10,10),
                translate_percent=(-0.05, 0.05),
                scale=(0.95,1.05),
                fill=0,
                border_mode=cv2.BORDER_CONSTANT,
                p = 0.5
            )
        ],
        bbox_params=A.BboxParams(
            format='pascal_voc',  # 1. 指定传入的格式
            label_fields=['category_ids'],  # 2. 告诉库对应的类别标签变量名
            min_visibility=0.3
        )
    )
    validation_transform = resize_and_pad()
    return train_transform, validation_transform
train_transform, valid_transform = make_transforms()
class UnderwaterData(Dataset):
    def __init__(self, path_image, path_label,train,augment=False):
        self.path_image = path_image
        self.path_label = path_label
        self.train = train
        self.augment = augment
    def __len__(self):
        return len(self.train)

    def __getitem__(self, item):
        box = []
        label = []
        extract_id = self.train[item]
        pic = cv2.imread(self.path_image + '/' + extract_id + '.jpg')
        pic = cv2.cvtColor(pic, cv2.COLOR_BGR2RGB)
        pic_h, pic_w, _ = pic.shape

        with open(self.path_label + '/' + extract_id + '.txt') as f:
            label_tra = f.readlines()
        for i in range(0, len(label_tra)):
            part = label_tra[i].strip().split()
            id = int(part[0])  # type
            cx = float(part[1]) * pic_w  # center x
            cy = float(part[2]) * pic_h  # center y
            wid = float(part[3]) * pic_w  # width
            heig = float(part[4]) * pic_h  # Height

            # standard of RCNN box is xmin,ymin,xmax,ymax
            x_min = cx - wid / 2
            y_min = cy - heig / 2
            x_max = cx + wid / 2
            y_max = cy + heig / 2
            box.append([x_min, y_min, x_max, y_max])
            label.append(id+1)

        pic_tensor = torch.from_numpy(pic).permute(2, 0, 1)
        pic_tensor = pic_tensor.float() / 255  # float 32, normalize
        if len(box) != 0:
            box = torch.as_tensor(box, dtype=torch.float32)
            label = torch.as_tensor(label, dtype=torch.int64)
        else:
            box = torch.zeros((0, 4), dtype=torch.float32)
            label = torch.zeros((0,), dtype=torch.int64)

        target = {
            'image_id': torch.tensor([item]),
            'boxes': box,
            'labels': label
        }
        if self.augment and random.random() < 0.5:
            pic_tensor = torch.flip(pic_tensor, dims=[2])  # flip width axis (C,H,W)
            if box.shape[0] > 0:
                flipped = box.clone()
                flipped[:, 0] = pic_w - box[:, 2]  # new xmin = W - old xmax
                flipped[:, 2] = pic_w - box[:, 0]  # new xmax = W - old xmin
                box = flipped
                target['boxes'] = box  # FIX: target 在翻转前已建好，原来只改了局部变量 box，标签框没跟着翻转
        return pic_tensor, target
