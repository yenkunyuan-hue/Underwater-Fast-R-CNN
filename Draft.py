# from torchvision.models import resnet50 as model
# from torchvision.models import ResNet50_Weights as weight
# import os
# import ssl
#
# ssl._create_default_https_context = ssl._create_unverified_context
# import numpy as np
# from sklearn.model_selection import KFold, train_test_split
# from torchvision.models.detection import FasterRCNN
# model = model(weights = weight.DEFAULT)
# print(model)
# path_img = '/Users/yangkunyuan/PycharmProjects/Underwater/UMOD/vis/img'
# path_label = '/Users/yangkunyuan/PycharmProjects/Underwater/UMOD/vis/labels'
# image = os.listdir(path_img)
# import torch
# from sklearn.metrics import  classification_report, confusion_matrix
# y_true = torch.tensor([0, 1, 2, 0, 1, 2, 0, 0]) # 真实类别 (0:猫, 1:狗, 2:鼠)
# y_pred = [0, 1, 1, 0, 1, 2, 0, 2] # 模型预测的类别
#
# # 方法 1：直接输出详细统计报告（包含每个类的 Precision, Recall, F1）
# target_names = ['Cat', 'Dog', 'Rat']
# print(classification_report(y_true, y_pred, target_names=target_names))
#
# # 方法 2：利用混淆矩阵手动计算每个类别的准确率 (Per-Class Accuracy)
# cm = confusion_matrix(y_true, y_pred)
# per_class_accuracy = cm.diagonal() / cm.sum(axis=1)
# # print(per_class_accuracy)
# from torchmetrics.detection.mean_ap import MeanAveragePrecision
# metric = MeanAveragePrecision(iou_type="bbox", class_metrics=True)
# import os
# from collections import Counter
#
#
# path_label = '/Users/yangkunyuan/PycharmProjects/Underwater/UMOD/vis/labels'
#
# # 1. 统计整个数据集的类别分布
# all_labels = []
# for label_file in os.listdir(path_label):
#     if label_file.endswith('.txt'):
#         with open(os.path.join(path_label, label_file)) as f:
#             for line in f.readlines():
#                 parts = line.strip().split()
#                 if parts:
#                     all_labels.append(int(parts[0]))

a = [1,3,3]
for c, id in enumerate (a):
    print(id)
    print(c)

