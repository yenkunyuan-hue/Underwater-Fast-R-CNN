import cv2
import torch
import random
from torch.utils.data import Dataset

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
        pic_tensor = torch.from_numpy(pic).permute(2, 0, 1)
        pic_tensor = pic_tensor.float() / 255  # float 32, normalize

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
        return pic_tensor, target
