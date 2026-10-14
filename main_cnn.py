from torch.utils.data import DataLoader
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from sklearn.model_selection import train_test_split
from model import CNN
import ssl
from plot.graph_plot import *
from data.data import UnderwaterData
from model.CNN import *
from loss.NMS import *
from loss.early_stop import if_early_stop
torch.cuda.empty_cache()

def one_epoch_train(model, train_loader, optimizer, device, validation_loader):
    running_loss = 0.00
    batch_count = 0
    total_batches = len(train_loader)
    model.train()
    model.set_drop_out(rate=0.1)
    loss_sum_dict = 0
    for images, targets in train_loader:
        images_batch = torch.stack(images, dim=0).to(device)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets] #load label and box into gpu
        loss_dict = model(images_batch,targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad(set_to_none=True)
        losses.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
        optimizer.step()
        running_loss += losses.item()
        batch_count += 1
    batch_valid_count = 0
    loss_box_valid = 0
    model.train()
    model.set_drop_out(rate=0)  # FIX: 验证 loss 用来做早停，关掉 dropout 减少噪声
    with torch.no_grad():
        for images, targets in validation_loader:
            images_batch = torch.stack(images, dim=0).to(device)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
            loss_dict_valid = model(images_batch, targets)
            loss_sum_dict += sum(loss for loss in loss_dict_valid.values()).item()
            batch_valid_count +=1
    return round(running_loss / batch_count,3), round(loss_sum_dict / batch_valid_count,3)


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
                {'params': model.model.stage1.parameters(), 'lr': 1e-5},
                {'params': model.model.stage2.parameters(), 'lr': 1e-5},
                {'params': base_params, 'lr': 4e-5},
                {'params': dcn_offset_params, 'lr': 4e-5}
            ], weight_decay=1e-6)
        print('解冻')
    return optimizer


if __name__ == "__main__":
    base_lr = 5e-5
    min_lr = 1e-6
    path_img = 'UMOD/vis/img'
    path_label = 'UMOD/vis/labels'

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
                                train=test_set,
                                augment=False)
    validation_set = UnderwaterData(path_label=path_label,
                                path_image=path_img,
                                train=validation_set,
                                augment=False)

    train_loader = DataLoader(
        train_data,
        batch_size = 8,
        shuffle=True,
        collate_fn=CNN.detection_collate_fn
    )
    test_loader = DataLoader(
        test_data,
        batch_size = 8,
    shuffle = False,
    collate_fn = CNN.detection_collate_fn
)
    validation_loader = DataLoader(
        validation_set,
        batch_size = 8,
        shuffle=False,
        collate_fn= detection_collate_fn
    )
    anchor_sizes = kmeans_anchor_sizes(path_label, train_set)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"当前计算加速设备: {device}")
    FCN_model = fcnn()
    model = detector(model=FCN_model, anchor_sizes = anchor_sizes).to(device)
    optimizer = set_optimizer(model, freeze=True)
    max_epoch = 100  # 早停没触发时的上限
    unfreeze_epoch = 10
    early_stopper = if_early_stop(patience=10)
    loss_history = [[], [], []]  # 0: epoch, 1: train loss, 2: validation loss

    for epoch in range(max_epoch):
        torch.cuda.empty_cache()
        print(f"Current Epoch{epoch}")
        train_loss, validation_loss = one_epoch_train(model, train_loader,
                                                        optimizer, device, validation_loader)
        print(f"train loss: {train_loss}, validation loss: {validation_loss}")
        loss_history[0].append(epoch)
        loss_history[1].append(train_loss)
        loss_history[2].append(validation_loss)
        if early_stopper(validation_loss, model, epoch):
            print(f"Early stop at epoch {epoch}, best epoch {early_stopper.best_epoch}, "
                  f"best validation loss {early_stopper.best_loss}")
            break
        if epoch+1 == unfreeze_epoch:
            model.set_trainable(True)
            optimizer = set_optimizer(model, freeze=False)
    early_stopper.load_best(model)  # FIX: 用验证 loss 最低那一轮的权重做测试
    prediction, target = testing(model, test_loader, device)
    loss_graph_plot(loss_history)