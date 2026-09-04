import torch
import torch.nn as nn
from torchvision.models import resnet50 as res_model, Weights
from torchvision.models import ResNet50_Weights
from torchvision.models._utils import IntermediateLayerGetter

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
        for param in self.backbone.parameters():
            param.requires_grad = trainable

class positional_encode(nn.Module):
        # make a X-Y coordinate to highlight the positional feature to the model
    def __init__(self, num_feature=256, temperature=10000):
            super().__init__()
            self.temperature = temperature
            self.num_feature = num_feature

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