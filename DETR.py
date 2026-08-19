import torch
from matplotlib.font_manager import weight_dict
from sympy.integrals.manualintegrate import trig_sincos_rule
from sympy.physics.units import temperature

from D_CNN_1 import UnderwaterData
import os
from sklearn.model_selection import KFold, train_test_split
import numpy as np
import torch.nn as nn
from torchvision.models import resnet50 as res_model, Weights
from torchvision.models import ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops import box_convert, generalized_box_iou_loss
import ssl
import torch.nn.functional as F
from matcher import build_matcher, HungarianMatcher

ssl._create_default_https_context = ssl._create_unverified_context

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
print(f"当前计算加速设备: {device}")


class DETR_backbone(nn.Module):
    def __init__(self, num_feature=256):
        super().__init__()
        resnet = res_model(weights=ResNet50_Weights.DEFAULT)
        return_layer = {
            'layer4': 'layer4'
        }
        self.num_feature = num_feature
        feature_resnet = IntermediateLayerGetter(resnet, return_layers=return_layer)
        self.backbone = feature_resnet
        self.input = nn.Conv2d(2048, num_feature, kernel_size=1 )
    def forward(self,data_in):
        data_out = self.backbone(data_in)['layer4']
        output_proj = self.input(data_out) #return is a dictionary
        return output_proj
    def freeze_layer(self, trainable=False):
        for param in self.resnet.parameters():
            param.requires_grad = trainable

class positional_encode(nn.Module):
    #make a X-Y coordinate to highlight the positional feature to the model
    def __init__(self,num_feature=256, temperature = 10000):
        super().__init__()
        self.temperature = temperature
        self.num_feature = num_feature

    class positional_encode(nn.Module):
        # make a X-Y coordinate to highlight the positional feature to the model
        def __init__(self, num_feature=256, temperature=10000):
            super().__init__()
            self.temperature = temperature
            self.num_feature = num_feature

        def forward(self, image):
            batch, channel, x, y = image.shape
            mask = torch.ones((batch, x, y), device=image.device)
            x_embed = mask.cumsum(1, dtype=torch.float32)
            # dim 指的是batch,y,z,根据哪个作为制作递增的依据
            y_embed = mask.cumsum(2, dtype=torch.float32)
            # divide into x-y plane, each plane is processed with cos and sin as these 2 are orthogonal

            # PE = sin/cos (pos/temperature ** (2i/d))
            # i: index of dimension, d: dimenisonal size
            # dim_t是括号内的东西
            dim_t = torch.arange(self.num_feature//2, dtype=torch.float32, device=image.device)  # num_feature个1's list
            dim_t = self.temperature ** (2 * (dim_t // 2) / (self.num_feature // 2))
            # 把x-y坐标转换成num_feature/2个三角函数曲线
            # i就是对应0-num_feature
            pos_x = x_embed[:, :, :, None] / dim_t
            pos_y = y_embed[:, :, :, None] / dim_t

            pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
            # 把两个(B,H,W,num_feature/2)合并成(B,H,W,num_feature/2，2)再合并成(B,H,W,num_feature)
            pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

            pos = torch.cat((pos_x, pos_y), dim=3).permute(0, 3, 1, 2)
            # 和backbone格式统一，第二位是num_feature
            return pos

    def forward(self, image):
        batch, channel, x, y = image.shape
        mask = torch.ones((batch, x, y), device = image.device)
        x_embed = mask.cumsum(1, dtype=torch.float32)
        #dim 指的是batch,y,z,根据哪个作为制作递增的依据
        y_embed = mask.cumsum(2, dtype=torch.float32)
        #divide into x-y plane, each plane is processed with cos and sin as these 2 are orthogonal

        #PE = sin/cos (pos/temperature ** (2i/d))
        #i: index of dimension, d: dimenisonal size
        #dim_t是括号内的东西
        dim_t = torch.arange(self.num_feature//2, dtype=torch.float32, device=image.device) #num_feature个1's list
        dim_t = self.temperature ** (2*(dim_t//2)/(self.num_feature//2))
        #把x-y坐标转换成num_feature/2个三角函数曲线
        #i就是对应0-num_feature
        pos_x = x_embed[:,:,:,None] / dim_t
        pos_y = y_embed[:,:,:,None] / dim_t

        pos_x = torch.stack((pos_x[:,:,:,0::2].sin(), pos_x[:,:,:,1::2].cos()),dim=4).flatten(3)
        #把两个(B,H,W,num_feature/2)合并成(B,H,W,num_feature/2，2)再合并成(B,H,W,num_feature)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_x,pos_y),dim=3).permute(0,3,1,2)
        #和backbone格式统一，第二位是num_feature
        return pos

class Multi_Layer_Percep(nn.Module):
    def __init__(self, input_dim, out_dim, num_layer, hid_dim):
        super().__init__()
        self.num_layer = num_layer
        hidden_layer = [hid_dim] * (num_layer-1)
        self.layers = nn.ModuleList(
            nn.Linear(k,v) for k, v in zip([input_dim]+hidden_layer, hidden_layer+[out_dim])
        )
        #list make by pytorch, can automatically load list into GPU
        #compare with nn.sequential, there is no 强制连接/顺序 between layer
    def forward(self, x):
        for index, layer in enumerate(self.layers):
            if index < self.num_layer - 1: #index starts with 0
                x = nn.functional.relu(layer(x))
            else:
                x = layer(x)
        return x


class DETR(nn.Module):
    def __init__(self, num_class = 10, num_q = 100, num_head = 8,
                 num_encode_layer=6, num_decode_layer=6, hid_dim=256):
        super().__init__()
        self.backbone = DETR_backbone(hid_dim)
        self.positional_encode = positional_encode(hid_dim)

        self.num_query = num_q
        self.num_head = num_head

        self.bbox = Multi_Layer_Percep(hid_dim, num_class, num_decode_layer, hid_dim)

        self.transformer = nn.Transformer(
            num_decoder_layers=num_decode_layer,
            num_encoder_layers=num_encode_layer,
            nhead=num_head,
            d_model=256,
            batch_first = True,
            dim_feedforward = 2048
            #after multihead, use to upsampling the feature set to extract the similarity in a higher dimension
        )

        self.query_embed = nn.Embedding(num_q, hid_dim)
        #make a learnable tensor with a size of num_q x hid_dim

        # prediction head
        self.class_head = nn.Linear(hid_dim, num_class)
        self.bbox = Multi_Layer_Percep(input_dim=hid_dim,
                                       out_dim=4, num_layer=4, hid_dim=hid_dim)

    def forward(self, input_tensor):
        feature_embed = self.backbone(input_tensor)
        pos_embed = self.positional_encode(feature_embed)

        batch_size, f_num, h, w = feature_embed.shape
        feature_map = feature_embed.flatten(2).permute(0,2,1)
        pos_map = pos_embed.flatten(2).permute(0, 2, 1)
        #(batch_size, sequence_length, embedding_dim)

        query_embed = self.query_embed.weight.unsqueeze(0).repeat(batch_size,1,1)
        #change the dimension from num_q x hid_dim to batch_size x num_q x hid_dim
        query_tensor = torch.zeros_like(query_embed)

        hidden_state = self.transformer(
            feature_map+pos_map,
            query_tensor+query_embed
        )
        pred_box = self.bbox(hidden_state).sigmoid()
        pred_type = self.class_head(hidden_state)
        prediction = {
            'pred_boxes':pred_box,
            'pred_logits':pred_type
        }
        return prediction
    def melt(self, trainable=False):
        self.backbone.freeze_layer(trainable)


class criterion(nn.Module):
    def __init__(self, matcher, class_num, weight_dict, back_coef):
        super().__init__()
        self.matcher = matcher
        self.class_num = class_num
        self.weighr_dict = weight_dict
        self.back_coef = back_coef
        self.initial_weight = torch.ones(class_num)
        self.initial_weight[-1] = back_coef
        self.register_buffer('initial weight', self.initial_weight)

    def forward(self, prediction, target):
        num_box = 0
        dice = self.matcher(prediction,target)
        pred_type = prediction['pred_logits']
        pred_box = prediction['pred_boxes']
        #class_num处object is background
        target_type = torch.full(pred_type[:2],self.class_num)
        for i, (index_i, index_j) in enumerate(dice):
            num_box += len(index_i)
            target_type[i][index_i] = target['labels'][index_j]

        loss_classif = F.cross_entropy(pred_type,target_type)

        num_box = max(num_box,1)
        loss_bbox = 0
        loss_giou = 0


        for i, (index_i, index_j) in enumerate(dice):
            box_match_pred = pred_box[i, index_i]
            box_match_target = target[i]['boxes'][index_j]
            loss_bbox += F.l1_loss(box_match_pred, box_match_pred, reduction='sum')
            loss_giou += generalized_box_iou_loss(box_match_target,box_match_pred)
            #after matcher, dimension of predict and target is same
        loss_bbox = loss_bbox/num_box
        loss_giou = loss_giou/num_box
        loss_overall ={
            'loss_bbox': loss_bbox,
            'loss_giou': loss_giou,
            'loss_classification': loss_classif
        }
        for key in loss_overall.keys():
            if key in self.weighr_dict:
                loss_overall[key] = loss_overall[key] * self.weight_dict[key]
        return sum(loss_overall.values())

def one_epoch_train(model, image, target):
    prediction = model(image)
    matcher = HungarianMatcher(
        cost_class = 1,
    cost_bbox = 1,
    cost_giou = 1
    )




if __name__ == '__main__':
    image = torch.rand(2,3,256,256)
    print(image.shape)
    model = DETR(num_class=10)
    print(model(image)['box'].shape)
    print('----------')
    print(model(image)['type'].shape)







