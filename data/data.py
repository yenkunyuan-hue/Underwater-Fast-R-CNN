import cv2
import torch
import random

from torch.utils.data import Dataset
import albumentations as A
# 数据集图片全部是 1280×720（宽×高），原来写的 IMAGE_SIZE = 512 与实际不符
IMAGE_W = 1280
IMAGE_H = 720
def make_transforms():
    """Preserve the full image; apply random deformation only during training."""

    def resize_and_pad():
        a = A.Compose([
            A.LongestMaxSize(
                max_size=IMAGE_W,  # 长边是宽
                interpolation=cv2.INTER_LINEAR)]
            +
            [A.PadIfNeeded(
                min_height=IMAGE_H,  # 不再 pad 成正方形，保持 16:9
                min_width=IMAGE_W,
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
            # 裁剪到图像范围内：albumentations 的 pascal_voc 格式遇到越界框会直接报错
            x_min = max(cx - wid / 2, 0)
            y_min = max(cy - heig / 2, 0)
            x_max = min(cx + wid / 2, pic_w)
            y_max = min(cy + heig / 2, pic_h)
            if x_max <= x_min or y_max <= y_min:  # 跳过退化框
                continue
            box.append([x_min, y_min, x_max, y_max])
            label.append(id+1)

        if self.augment:
            # 用 make_transforms 里的 train_transform 同时变换图像和框，
            # min_visibility 过滤掉的框，其类别也会从 category_ids 里一起去掉
            augmented = train_transform(image=pic, bboxes=box, category_ids=label)
            pic = augmented['image']
            box = [list(b) for b in augmented['bboxes']]
            label = list(augmented['category_ids'])
        else:
            augmented = valid_transform(image=pic, bboxes=box, category_ids=label)
            pic = augmented['image']
            box = [list(b) for b in augmented['bboxes']]
            label = list(augmented['category_ids'])
        # 验证/测试集不做变换：图片本身就是 1280×720，valid_transform 对它们等价于不变；
        # graph_plot 在原图上画框、k-means anchor 也按 1280×720 算，保持原尺寸才能对得上
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
        return pic_tensor, target
