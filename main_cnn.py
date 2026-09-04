from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.model_selection import train_test_split
from model import CNN
import ssl
from plot.graph_plot import *
from data.data import UnderwaterData
from model.CNN import *
from loss.NMS import *
torch.cuda.empty_cache()

def one_epoch_train(model, train_loader, optimizer, device, validation_loader):
    running_loss = 0.00
    batch_count = 0
    total_batches = len(train_loader)
    model.train()
    model.set_drop_out(rate=0.1)
    loss_sum_dict = 0
    loss_box = 0
    for images, targets in train_loader:
        images_batch = torch.stack(images, dim=0).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] #load label and box into gpu
        loss_dict = model(images_batch,targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        loss_box += loss_dict['loss_box_reg'].item()
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
    return round(running_loss / batch_count,3), round(loss_box / batch_count, 3), round(loss_sum_dict / batch_valid_count,3), round(loss_box_valid / batch_valid_count, 3)


def testing(model, data_loader, device):
    metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
    model.eval()
    model.set_drop_out(rate=0)
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
                label_predict = (predictions_cpu[i]['labels'].numpy()) - 1
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
        object_names = ['cage', 'frame', 'hook', 'anchor', 'tire', 'rov', 'plastic bucket', 'fish', 'oil drum']

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
        return  result_dict, target_dict


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
        {'params': dcn_offset_params, 'lr': 8e-5}  # 探测支路双倍速学习
    ], weight_decay=1e-6)
    else:
        optimizer = torch.optim.AdamW([
                {'params': model.model.stage1.parameters(), 'lr': 1e-6},
                {'params': model.model.stage2.parameters(), 'lr': 1e-6},
                {'params': base_params, 'lr': 4e-5},
                {'params': dcn_offset_params, 'lr': 4e-5}
            ], weight_decay=1e-6)
        print('解冻')
    return optimizer

epoch_num = 60
precision_history = np.zeros((5, epoch_num))

#0: Average Loss, 1: Box Loss, 2: AL for validation, 3:Box for valid
if __name__ == "__main__":
    # path_img = '/Users/yangkunyuan/PycharmProjects/Underwater/UMOD/vis/img'
    # path_label = '/Users/yangkunyuan/PycharmProjects/Underwater/UMOD/vis/labels'
    base_lr = 5e-5
    min_lr = 1e-5
    warmup_epochs = 10
    path_img = '/ceph/home/student.aau.dk/fy14pi/Underwater/UMOD/vis/img'
    path_label = '/ceph/home/student.aau.dk/fy14pi/Underwater/UMOD/vis/labels'

    img = os.listdir(path_img)
    for i in range(0, len(img)):
        img[i], _ =os.path.splitext(img[i])
    img = np.array(img)
    train_set, test_set = train_test_split(img, shuffle=True, random_state=20, test_size=0.1)
    train_set, validation_set = train_test_split(train_set,shuffle=True, random_state=20, test_size=0.1)
    train_data = UnderwaterData(path_label=path_label,
                                path_image=path_img,
                                train=train_set,
                                augment=True)
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
        collate_fn=CNN.detection_collate_fn
    )
    test_loader = DataLoader(
        test_data,
        batch_size = 2,
    shuffle = False,
    collate_fn = CNN.detection_collate_fn
)
    validation_loader = DataLoader(
        validation_set,
        batch_size = 2,
        shuffle=False,
        collate_fn= detection_collate_fn
    )
    anchor_sizes = kmeans_anchor_sizes(path_label, train_set)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"当前计算加速设备: {device}")
    FCN_model = fcnn()
    model = detector(model=FCN_model, anchor_sizes = anchor_sizes).to(device)
    optimizer = set_optimizer(model, freeze=True)

    for epoch in range(epoch_num):
        torch.cuda.empty_cache()
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
        if epoch+1 == 45:
            model.set_trainable(True)
            optimizer = set_optimizer(model, freeze=False)
    prediction, target = testing(model, test_loader, device)
    #prediction = NMS_with_label(prediction=prediction, IOU_threshold=0.5)
    graph_plot(path_image=path_img, prediction=prediction, round=epoch_num, target=target)
    loss_graph_plot(precision_history)