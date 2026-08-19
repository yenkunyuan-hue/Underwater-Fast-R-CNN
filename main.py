import cv2
import torch
import torch.nn as nn
from sklearn.metrics import precision_score, accuracy_score
from sklearn.utils.multiclass import is_multilabel
from torch.nn.functional import dropout
from torchvision.models import vgg16 as model
from torchvision.models import VGG16_Weights
from torch.utils.data import Dataset, DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision.ops as ops
import torchvision.utils as vutils
import torchvision.transforms.functional as F
import torch.nn.functional as f
import sklearn
import os
import numpy as np
from sklearn.model_selection import KFold, train_test_split
import matplotlib.pyplot as plt
import ssl
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.anchor_utils import AnchorGenerator
from sklearn.metrics import classification_report
from torchmetrics.regression import MeanSquaredError, MeanAbsolutePercentageError, MeanAbsoluteError

ssl._create_default_https_context = ssl._create_unverified_context

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


class UnderwaterData(Dataset):
    def __init__(self, path_image, path_label,train):
        self.path_image = path_image
        self.path_label = path_label
        self.train = train
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
        return pic_tensor, target

# selec_box('vis-0010')
# targets = label_ex('vis-0010')
#picture = image_ex('vis-0010')
class fcnn(nn.Module):
    def __init__(self):
        super().__init__()
        #load feature extractor
        vgg16 = model(weights = VGG16_Weights.DEFAULT)
        vgg16 = vgg16.features
        vgg16_layers = list(vgg16.children())
        vgg16_layers[24] = DCN(input_channel=512,output_channel=512, kernel_size=3, stride=(1, 1), padding=(1, 1))
        vgg16_layers[26] = DCN(input_channel=512,output_channel=512, kernel_size=3, stride=(1, 1), padding=(1, 1))
        vgg16_layers[28] = DCN(input_channel=512,output_channel=512, kernel_size=3, stride=(1, 1), padding=(1, 1))

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

        feature_lateral3_sam = f.interpolate(feature_lateral3, scale_factor=2, mode='nearest')
        feature_lateral2 = feature_lateral2 + feature_lateral3_sam  # 改为 + 非原地加法

        feature_lateral2_sam = f.interpolate(feature_lateral2, scale_factor=2, mode='nearest')
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

def detection_collate_fn(batch):
    return tuple(zip(*batch)) #多目标拆开来，默认的只接受单目标的tensor

class detector(nn.Module):
    def __init__(self,model,num_class = 10, drop_out_rate=0.1):
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
                sizes=((64,), (128,),(256,)),
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
    def forward(self,images_batch, targets=None):
        return self.detector(images_batch, targets)
    def set_drop_out(self, rate:float):
        self.new_box_head.set_drop_out(rate)
    def set_trainable(self, trainable:bool):
        self.model.stage_trainable(trainable = trainable)

def one_epoch_train(model, train_loader, optimizer, device, validation_loader):
    running_loss = 0.00
    batch_count = 0
    total_batches = len(train_loader)
    model.train()
    model.set_drop_out(rate=0.2)
    loss_sum_dict = 0
    loss_box = 0
    for images, targets in train_loader:
        images_batch = torch.stack(images, dim=0).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] #load label and box into gpu
        loss_dict = model(images_batch,targets)
        losses = sum(loss for loss in loss_dict.values())
        loss_box += loss_dict['loss_box_reg'].item()
        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        running_loss += losses.item()
        batch_count += 1
    batch_valid_count = 0
    loss_box_valid = 0
    model.train()
    model.set_drop_out(rate=0)
    with torch.no_grad():
        for images, targets in validation_loader:
            images_batch = torch.stack(images, dim=0).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict_valid = model(images_batch, targets)
            loss_sum_dict += sum(loss for loss in loss_dict_valid.values()).item()
            loss_box_valid += loss_dict_valid['loss_box_reg'].item()
            batch_valid_count +=1
    return running_loss / batch_count, loss_box / batch_count, loss_sum_dict / batch_valid_count, loss_box_valid / batch_valid_count
def testing(model, data_loader, device):
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    model.eval()
    result_dict = {}
    target_dict={}
    target_correct_sum = []
    predict_correct_sum = []
    with torch.no_grad():
        for images, targets in data_loader:
            images_batch = torch.stack(images, dim=0).to(device)
            predictions = model(images_batch)
            targets_cpu = [{k: v.to('cpu') for k, v in t.items()} for t in targets]
            predictions_cpu = [{k: v.to('cpu') for k, v in p.items()} for p in predictions]

            #Update label, as 0 means ground in the model
            for t, p in zip(targets_cpu,predictions_cpu):
                target_correct = t.copy()
                target_correct['labels'] =target_correct['labels'] - 1
                target_correct_sum.append(target_correct)

                predict_correct = p.copy()
                predict_correct['labels'] = predict_correct['labels'] - 1
                predict_correct_sum.append(predict_correct)

            for i in range(0,len(predictions_cpu)):
                box_predict = (predictions_cpu[i]['boxes'].numpy())
                label_predict = (predictions_cpu[i]['labels'].numpy()) -1
                confidence_predict = (predictions_cpu[i]['scores'].numpy())
                image_id = targets_cpu[i]['image_id'].item()
                result_dict[image_id] = {
                'box': box_predict,
                'label': label_predict,
                'confidence':confidence_predict
            }
                box_target = (targets_cpu[i]['boxes'].numpy())
                label_target = (targets_cpu[i]['labels'].numpy()) - 1
                target_dict[image_id] = {
                        'boxes': box_target,
                        'labels': label_target,
                    }
        metric.update(predict_correct_sum, target_correct_sum)
        result = metric.compute()

        print("\n================ 测试集评估结果 ================")
        print(f"mAP (IoU=0.50:0.95): {result['map'].item():.4f}")  # 综合大指标
        print(f"mAP_50 (IoU=0.50)  : {result['map_50'].item():.4f}")  # 经典 PASCAL VOC 指标
        print(f"mAP_75 (IoU=0.75)  : {result['map_75'].item():.4f}")  # 严格指标
        print("================================================\n")
        print(f"mAP Per Class:")
        object_names = ['cage', 'frame', 'hook', 'anchor', 'tire', 'rov', 'fish', 'oil drum']

        map_per_class = result['map_per_class'].tolist()
        mar_per_class = result['mar_100_per_class'].tolist()
        print(mar_per_class)

        print("\n================ Per-Class Metrics ================")
        # 表头：<18 表示左对齐占用18字符宽度，>10 表示右对齐占用10字符宽度
        print(f"{'Class Name':<18} | {'mAP':>10} | {'Recall (mar_100)':>18}")
        print("-" * 55)

        for name, map_val, mar_val in zip(object_names, map_per_class, mar_per_class):
            # 处理 -1.0 (代表测试集中无该类真实样本的情况)
            map_str = f"{map_val:.4f}" if map_val != -1.0 else "N/A"
            mar_str = f"{mar_val:.4f}" if mar_val != -1.0 else "N/A"

            print(f"{name:<18} | {map_str:>10} | {mar_str:>18}")

        print("===================================================================\n")
        return  result_dict, target_dict

def graph_plot(path_image, target, prediction, round):

    object_name = ['cage', 'frame', 'hook', 'anchor', 'tire', 'rov', 'plastic bucket', 'fish', 'oil drum']
    colour_predict_box = (0, 255, 255)  # 亮黄色
    colour_predict_text = (0, 255, 255)  # 亮黄色
    colour_target_box = (255, 0, 255)  # 亮洋红/紫粉色
    colour_target_text = (255, 0, 255)  # 亮洋红/紫粉色

    save_path = '/ceph/home/student.aau.dk/fy14pi/test_picture'
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs('test_picture')

    for image_id in prediction:
        save_path = '/ceph/home/student.aau.dk/fy14pi/test_picture'
        read_path = test_set[image_id]
        image = cv2.imread(path_image+'/'+read_path+'.jpg')
        image = image.copy()
        prediction_img = prediction[image_id]
        target_img = target[image_id]
        for i in range (0,len(prediction_img['box'])):
            if len(prediction_img['box']) == 0:
                break
            xmin, ymin, xmax, ymax = map(int, prediction_img['box'][i])
            text = f"{object_name[int(prediction_img['label'][i])]}_{prediction_img['confidence'][i]:.2f}"
            if float(prediction_img['confidence'][i]) >= 0.1:
                cv2.rectangle(image,
                      (xmin,ymin),
                      (xmax,ymax),
                      color=colour_predict_box,
                      thickness=2)
                cv2.putText(image,
                    text,
                    (xmin + 50, ymin + 50),
                    color=colour_predict_text,
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                    fontScale=0.5,)
        for j in range(0,len(target_img['boxes'])):
            xmin, ymin, xmax, ymax = map(int, target_img['boxes'][j])
            text = 'Target: ' + str(object_name[int(target_img['labels'][j])])
            cv2.rectangle(image,
                          (xmin, ymin),
                          (xmax, ymax),
                          color=colour_target_box,
                          thickness=2)
            cv2.putText(image,
                        text,
                        (xmin +  50, ymin + 50),
                        color=colour_target_text,
                        fontFace=cv2.FONT_HERSHEY_SIMPLEX,  # 字体
                        fontScale=0.5, )
        read_path = read_path+'R'+str(round+1)+'.jpg'
        save_path = os.path.join(save_path,read_path)
        cv2.imwrite(save_path,img=image)

def NMS_with_label(prediction, IOU_threshold):
    # print(prediction)
    for image_id in prediction:
        prediction_pic = prediction[image_id]
        order_keep=[]
        box = np.array(prediction_pic['box'])
        label = np.array(prediction_pic['label'])
        score = np.array(prediction_pic['confidence'])
        box_max = box.max() if len(box) >0 else 0
        offset = label.astype(float) * (box_max + 1)
        box_offset = box + offset[:,None]
        xmin = box_offset[:,0]
        ymin = box_offset[:,1]
        xmax = box_offset[:,2]
        ymax = box_offset[:,3]
        area = (xmax-xmin) * (ymax - ymin)
        order = score.argsort()[::-1]
        while len(order) > 0:
            index_max = order[0]
            order_new = []
            order_keep.append(index_max)

            if len(order) == 1:
                break

            x_min_interset = np.maximum(xmin[index_max], xmin[order[1:]])
            y_min_intersect = np.maximum(ymin[index_max], ymin[order[1:]])
            x_max_intersect = np.minimum(xmax[index_max], xmax[order[1:]])
            y_max_intersect = np.minimum(ymax[index_max], ymax[order[1:]])
            width = np.maximum(0,x_max_intersect-x_min_interset)
            height = np.maximum(0,y_max_intersect-y_min_intersect)

            intersection_area = (width) * (height)
            IOU = intersection_area/(area[index_max] - intersection_area + area[order[1:]])

            for k in range(0,len(IOU)):
                if IOU[k] <= IOU_threshold:
                    order_new.append(order[k+1])  #0用来作为area考虑了，剩下考虑的都是n+1
            order = order_new
        prediction[image_id] = {
            'box': box[order_keep],
            'label': label[order_keep],
            'confidence': score[order_keep]
        }
        # print(prediction)

    return prediction

def loss_graph_plot(loss_history):
    save_path = '/ceph/home/student.aau.dk/fy14pi/loss_overall'
    if os.path.exists(save_path):
        pass
    else:
        os.makedirs('loss_overall')
    save_path_1 = '/ceph/home/student.aau.dk/fy14pi/loss_overall/Average_Loss.png'
    save_path_2 = '/ceph/home/student.aau.dk/fy14pi/loss_overall/Box_Loss.png'
    x1 = loss_history[0]
    y1 = (loss_history[1])
    y2 = (loss_history[2])
    y3 = (loss_history[3])
    y4 = (loss_history[4])
    plt.figure()
    plt.plot(x1,y1, label = 'Trainning Loss', color = 'blue')
    plt.plot(x1,y3, label = 'Validation Loss', color = 'red')
    plt.title('Average Loss Over Epoch')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(
        loc='upper right',  # 锁定右上角
        frameon=True,  # 开启图例外框
        facecolor='white',  # 图例背景设为白色
        framealpha=0.9,  # 背景透明度
        fontsize=11
    )
    plt.savefig(save_path_1)
    plt.close()

    plt.figure()
    plt.plot(x1, y2, label='Trainning Loss', color='blue')
    plt.plot(x1, y4, label='Validation Loss', color='red')
    plt.title('Box Loss Over Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend(
        loc='upper right',  # 锁定右上角
        frameon=True,  # 开启图例外框
        facecolor='white',  # 图例背景设为白色
        framealpha=0.9,  # 背景透明度
        fontsize=11
    )
    plt.savefig(save_path_2)
    plt.close()

epoch_num = 70
precision_history = np.zeros((5, epoch_num))

def set_optimizer(model, freeze = True):
    dcn_offset_params =[]
    base_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if "offset" in name:
                dcn_offset_params.append(param)  # 抓住你 DCN 里的 self.offset
            elif not freeze and ("stage1" in name or "stage2" in name):
                continue
            else:
                base_params.append(param)
    if freeze:
        optimizer = torch.optim.AdamW([
        {'params': base_params, 'lr': 4e-5},
        {'params': dcn_offset_params, 'lr': 4e-5}  # 探测支路双倍速学习
    ], weight_decay=1e-5)
    else:
        optimizer = torch.optim.AdamW([
                {'params': model.model.stage1.parameters(), 'lr': 5e-6},
                {'params': model.model.stage2.parameters(), 'lr': 5e-6},
                {'params': base_params, 'lr': 2e-5},
                {'params': dcn_offset_params, 'lr': 2e-5}
            ], weight_decay=1e-5)
        print('解冻')
    return optimizer
#0: Average Loss, 1: Box Loss, 2: AL for validation, 3:Box for valid
if __name__ == "__main__":
    # path_img = '/Users/yangkunyuan/PycharmProjects/Underwater/UMOD/vis/img'
    # path_label = '/Users/yangkunyuan/PycharmProjects/Underwater/UMOD/vis/labels'
    path_img = '/ceph/home/student.aau.dk/fy14pi/Underwater/UMOD/vis/img'
    path_label = '/ceph/home/student.aau.dk/fy14pi/Underwater/UMOD/vis/labels'


    img = os.listdir(path_img)
    for i in range(0, len(img)):
        img[i], _ = os.path.splitext(img[i])
    img = np.array(img)
    train_set, test_set = train_test_split(img, shuffle=True, random_state=20, test_size=0.1)
    train_set, validation_set = train_test_split(train_set,shuffle=True, random_state=20, test_size=0.1)
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
        batch_size = 2,
        shuffle=True,
        collate_fn=detection_collate_fn
    )
    test_loader = DataLoader(
        test_data,
        batch_size = 2,
    shuffle = False,
    collate_fn = detection_collate_fn
)
    validation_loader = DataLoader(
        validation_set,
        batch_size = 2,
        shuffle=False,
        collate_fn= detection_collate_fn
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"当前计算加速设备: {device}")
    FCN_model = fcnn()
    model = detector(model=FCN_model).to(device)
    optimizer = set_optimizer(model, freeze=True)

    for epoch in range(epoch_num):
        print(f"Current Epoch{epoch}")
        precision_history[0][epoch] = epoch
        (precision_history[1][epoch],
         precision_history[2][epoch],
         precision_history[3][epoch],
         precision_history[4][epoch]) = one_epoch_train(model, train_loader,
                                                        optimizer, device, validation_loader)
        if (epoch+1)//10 == ((epoch+1)/10) and (epoch+1) != 1:
             prediction,target = testing(model,test_loader,device)
             prediction = NMS_with_label(prediction=prediction, IOU_threshold=0.5)
             graph_plot(path_image=path_img,prediction=prediction,round=epoch, target = target)
        if epoch+1 == 15:
            model.set_trainable(True)
            optimizer = set_optimizer(model, freeze=False)
    prediction, target = testing(model, test_loader, device)
    #prediction = NMS_with_label(prediction=prediction, IOU_threshold=0.5)
    graph_plot(path_image=path_img, prediction=prediction, round=epoch_num, target=target)
    loss_graph_plot(precision_history)