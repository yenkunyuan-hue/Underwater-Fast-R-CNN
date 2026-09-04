from matplotlib.font_manager import weight_dict
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import os
from sklearn.model_selection import train_test_split
import numpy as np
from torch.utils.data import DataLoader
from main_cnn import UnderwaterData
import ssl
from loss.matcher import HungarianMatcher
from model import DETR
from loss.loss_function import *

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
torch.cuda.empty_cache()

print(f"当前计算加速设备: {device}")

def one_epoch_train(model, optimizer, train_loader,validation_loader, criterion):
    loss_overall_train = 0
    loss_overall_target = 0
    model.train()
    for image, target in train_loader:
        image_batch = torch.stack(image, dim=0).to(device)
        image_batch = image_batch.to(device)
        target = [{k: v.to(device) for k, v in t.items()} for t in target]
        prediction = model(image_batch)
        losses_train = criterion(prediction, target)
        loss = sum(losses_train.values())
        loss.backward()
        loss_overall_train += loss
        optimizer.zero_grad()
        optimizer.step()
    with torch.no_grad():
        for image, target in validation_loader:
            image_batch = torch.stack(image,dim=0).to(device)
            image_batch = image_batch.to(device)
            for t in target:
                for key, value in t.items():
                    t[key] = value.to(device)
            prediction = model(image_batch)
            loss_overall_target += sum(criterion(prediction, target).values())
    # return loss_overall_train, loss_overall_target
    return 1, 2
def detection_collate_fn(batch):
    return tuple(zip(*batch)) #多目标拆开来，默认的只接受单目标的tensor

def testing(model, data_loader, device):
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    target_correct_sum = []
    predict_correct_sum = []
    with torch.no_grad():
        for images, targets in data_loader:
            images_batch = torch.stack(images, dim=0).to(device)
            predictions = model(images_batch)
            targets_cpu = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
            prob = predictions['pred_logits'].softmax(-1)
            scores, labels = prob[..., 1:].max(-1)  # 选出 0~8 范围内的物体类
            labels += 1
            boxes = predictions['pred_boxes']

            #Update label, as 0 means ground in the model
            for t in (targets_cpu):
                target_correct = t.copy()
                target_correct_sum.append(target_correct)
            for i in range (0,len(targets_cpu)):
                predict_correct_sum.append({
                    'boxes': boxes[i].cpu(),
                    'scores': scores[i].cpu(),
                    'labels': (labels[i]).cpu()
                })
        metric.update(predict_correct_sum, target_correct_sum)
        result = metric.compute()

        print("\n================ 测试集评估结果 ================")
        print(f"mAP (IoU=0.50:0.95): {result['map'].item():.4f}")  # 综合大指标
        print(f"mAP_50 (IoU=0.50)  : {result['map_50'].item():.4f}")  # 经典 PASCAL VOC 指标
        print(f"mAP_75 (IoU=0.75)  : {result['map_75'].item():.4f}")  # 严格指标
        print("================================================\n")
        print(f"mAP Per Class:")
        object_names = ['background', 'cage', 'frame', 'hook', 'anchor', 'tire', 'rov', 'plastic bucket', 'fish', 'oil drum']

        map_per_class = result['map_per_class'].tolist()
        mar_per_class = result['mar_100_per_class'].tolist()
        name_list = result['classes'].tolist()

        print("\n================ Per-Class Metrics ================")
        # 表头：<18 表示左对齐占用18字符宽度，>10 表示右对齐占用10字符宽度
        print(f"{'Class Name':<18} | {'mAP':>10} | {'Recall (mar_100)':>18}")
        print("-" * 55)

        for idx, name_idx in enumerate(name_list):
            map_val = map_per_class[idx]
            mar_val = mar_per_class[idx]
            name = object_names[name_idx]
            # 处理 -1.0 (代表测试集中无该类真实样本的情况)
            map_str = f"{map_val:.2f}" if map_val != 0.0 else "N/A"
            mar_str = f"{mar_val:.2f}" if mar_val != 0.0 else "N/A"

            print(f"{name:<18} | {map_str:>10} | {mar_str:>18}")

        print("===================================================================\n")


path_img = '/ceph/home/student.aau.dk/fy14pi/Underwater/UMOD/vis/img'
path_label = '/ceph/home/student.aau.dk/fy14pi/Underwater/UMOD/vis/labels'

img = os.listdir(path_img)
for i in range(0, len(img)):
    img[i], _ = os.path.splitext(img[i])
img = np.array(img)
train_set, test_set = train_test_split(img, shuffle=True, random_state=20, test_size=0.1)
train_set, validation_set = train_test_split(train_set, shuffle=True, random_state=20, test_size=0.1)
train_data = UnderwaterData(path_label=path_label,
                                path_image=path_img,
                                train=train_set)
test_data = UnderwaterData(path_label=path_label,
                               path_image=path_img,
                               train=test_set)
validation_set = UnderwaterData(path_label=path_label,
                                    path_image=path_img,
                                    train=validation_set)

train_loader = DataLoader(
        train_data,
        batch_size=2,
        shuffle=True,
        collate_fn=detection_collate_fn
    )

test_loader = DataLoader(
        test_data,
        batch_size=2,
        shuffle=False,
        collate_fn=detection_collate_fn
    )
validation_loader = DataLoader(
        validation_set,
        batch_size=2,
        shuffle=False,
        collate_fn=detection_collate_fn
    )
device = torch.device(
        "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"当前计算加速设备: {device}")

matcher = HungarianMatcher(
        cost_class = 1,
    cost_bbox = 1,
    cost_giou = 1
    )
model = DETR.DETR()
model = model.to(device)
criterion = criterion(matcher=matcher,weight_dict=weight_dict)
criterion = criterion.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr= 5e-6)

if __name__ == '__main__':
    epoch_num = 100
    for epoch in range(0,epoch_num):
        torch.cuda.empty_cache()
        train_loss, valid_loss = one_epoch_train(criterion=criterion,
                                                 model=model, optimizer=optimizer,
                                                 train_loader=train_loader,
                                                 validation_loader = validation_loader)

        print(f'train loss in epoch{epoch}: {train_loss}')
        print(f'validation loss in epoch{epoch}: {valid_loss}')
        if epoch%10 == 0:
            testing(model,test_loader,device)






