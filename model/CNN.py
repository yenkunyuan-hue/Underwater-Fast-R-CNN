import torch
import torch.nn as nn
from torchvision.models import vgg16 as model
from torchvision.models import VGG16_Weights
import torchvision.ops as ops
import torch.nn.functional as f
import os
import numpy as np
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from model.S3_DSConv import DSConv
from loss.loss_function import CIoURoIHeads
device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")

class FC_with_dropout(nn.Module):
    def __init__(self, in_channels, representation_size, dropout_rate = 0.1):
        super().__init__()
        self.out_channels = representation_size
        self.fc6 = nn.Linear(in_channels, representation_size)
        self.fc7 = nn.Linear(representation_size, representation_size)
        self.dropout = nn.Dropout(p=dropout_rate)
    def forward(self,x):
        x = x.flatten(start_dim=1)
        # FC6 -> ReLU -> Dropout
        x = f.relu(self.fc6(x))
        x = self.dropout(x)
        # FC7 -> ReLU -> Dropout
        x = f.relu(self.fc7(x))
        x = self.dropout(x)
        return x
    def set_drop_out(self, rate:float):
        self.dropout.p = rate


# selec_box('vis-0010')
# targets = label_ex('vis-0010')
#picture = image_ex('vis-0010')
class DSConv_2D(nn.Module):
    def __init__(self, out_channel, int_channel):
        super().__init__()
        self.out_channel = out_channel
        self.x = DSConv(in_ch=int_channel,
        out_ch=out_channel//2,
        kernel_size=15,
        extend_scope=1,
        morph=0,
        if_offset=True,
        device=device)
        self.y = DSConv(in_ch=int_channel,
        out_ch=out_channel//2,
        kernel_size=15,
        extend_scope=1,
        morph=1,
        if_offset=True,
        device=device)
        self.fuse_channel = nn.Conv2d(kernel_size=1, in_channels=self.out_channel, out_channels=self.out_channel, stride=1, padding=0)
    def forward(self, image):
        x = self.x(image)
        y = self.y(image)
        output = torch.cat((x,y),dim=1) #concate along the channel
        output_fused_channel = self.fuse_channel(output)
        return output_fused_channel




class fcnn(nn.Module):
    def __init__(self):
        super().__init__()
        #load feature extractor
        vgg16 = model(weights = VGG16_Weights.DEFAULT)
        vgg16 = vgg16.features
        vgg16_layers = list(vgg16.children())

        vgg16_layers[24] = DSConv_2D(int_channel=512,out_channel=512)
        vgg16_layers[26] = DSConv_2D(int_channel=512,out_channel=512)
        vgg16_layers[28] = DSConv_2D(int_channel=512,out_channel=512)

        vgg_new = nn.Sequential(*vgg16_layers[0:31])

        self.stage1 = nn.Sequential(*vgg16_layers[0:17])
        self.stage2 = nn.Sequential(*vgg16_layers[17:24])
        self.stage3 = nn.Sequential(*vgg16_layers[24:31])

        for param in self.stage2.parameters():
            param.requires_grad = False
        for param in self.stage1.parameters():
            param.requires_grad = False

        self.lateral1 = nn.Conv2d(256, 256, kernel_size=1)
        self.lateral2 = nn.Conv2d(512, 256, kernel_size=1)
        self.lateral3 = nn.Conv2d(512, 256, kernel_size=1)

        self.smooth_1 = nn.Conv2d(256,256,kernel_size=3, padding=1)
        self.smooth_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.smooth_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
    def forward_feature(self,image):
        c1 = self.stage1(image)
        c2 = self.stage2(c1)
        c3 = self.stage3(c2)

        feature_lateral1 = self.lateral1(c1)
        feature_lateral2 = self.lateral2(c2)
        feature_lateral3 = self.lateral3(c3)

        feature_lateral3_sam = f.interpolate(feature_lateral3, size=feature_lateral2.shape[2:], mode='nearest')
        feature_lateral2 = feature_lateral2 + feature_lateral3_sam  # 改为 + 非原地加法

        feature_lateral2_sam = f.interpolate(feature_lateral2, size=feature_lateral1.shape[2:], mode='nearest')
        feature_lateral1 = feature_lateral1 + feature_lateral2_sam  # 改为 + 非原地加法

        feature_lateral1 = self.smooth_1(feature_lateral1)
        feature_lateral2 = self.smooth_2(feature_lateral2)
        feature_lateral3 = self.smooth_3(feature_lateral3)

        return feature_lateral1, feature_lateral2, feature_lateral3

    def stage_trainable(self, trainable:bool):
        for param in self.stage2.parameters():
            param.requires_grad = trainable
        for param in self.stage1.parameters():
            param.requires_grad = trainable

class DCN(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size, stride, padding):
        super().__init__()
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.offset = nn.Conv2d(in_channels=input_channel,
                                out_channels=2*kernel_size*kernel_size,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride)
        nn.init.constant_(self.offset.weight, 0)
        nn.init.constant_(self.offset.bias, 0)
        self.deform = ops.DeformConv2d(
            in_channels=input_channel,
            out_channels=output_channel,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )
    def forward(self,image):
        offset_v = self.offset(image)
        result = self.deform(image,offset_v)
        return result

def kmeans_anchor_sizes(path_label, image_ids, img_w=1280, img_h=720, k=3, n_iter=100, seed=42):
    sizes = []
    for sid in image_ids:
        label_path = os.path.join(path_label, sid + '.txt')
        if not os.path.exists(label_path):
            continue
        with open(label_path) as label_file:
            for line in label_file:
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                w = float(parts[3]) * img_w
                h = float(parts[4]) * img_h
                if w > 0 and h > 0:
                    sizes.append((w * h) ** 0.5)

    sizes = np.array(sizes)
    if len(sizes) < k:
        return (64, 128, 256)  # not enough boxes to cluster meaningfully

    rng = np.random.default_rng(seed)
    centers = rng.choice(sizes, size=k, replace=False)
    for _ in range(n_iter):
        distances = np.abs(sizes[:, None] - centers[None, :])
        assignments = distances.argmin(axis=1)
        new_centers = centers.copy()
        for i in range(k):
            members = sizes[assignments == i]
            if len(members) > 0:
                new_centers[i] = members.mean()
        converged = np.allclose(new_centers, centers)
        centers = new_centers
        if converged:
            break
    return tuple(int(round(c)) for c in sorted(centers))


def detection_collate_fn(batch):
    return tuple(zip(*batch)) #多目标拆开来，默认的只接受单目标的tensor

class detector(nn.Module):
    def __init__(self,model,num_class = 10, drop_out_rate=0.1,anchor_sizes=(64, 128, 256)):
        super().__init__()
        self.model = model
        self.model.out_channels = 256
        def forward_bridge(image):
            current_device = next(self.model.parameters()).device
            image = image.to(current_device)
            p3, p4, p5 = self.model.forward_feature(image)
            return {'0': p3, '1': p4, '2': p5}
        self.model.forward = forward_bridge
        anchor = AnchorGenerator(
            sizes=tuple((s,) for s in anchor_sizes),
            aspect_ratios=((0.5, 1.0, 2.0),) * 3
        )
        self.new_box_head = FC_with_dropout(
            in_channels=256 * 7 * 7,
            representation_size=1024,
        )
        self.detector = FasterRCNN(
            backbone=self.model,
            num_classes=num_class,
            rpn_anchor_generator=anchor,
            box_nms_thresh= 0.1,
            box_score_thresh= 0.4,
            box_head = self.new_box_head
        )
        self.detector.roi_heads.__class__ = CIoURoIHeads
    def forward(self,images_batch, targets=None):
        return self.detector(images_batch, targets)
    def set_drop_out(self, rate:float):
        self.new_box_head.set_drop_out(rate)
    def set_trainable(self, trainable:bool):
        self.model.stage_trainable(trainable = trainable)