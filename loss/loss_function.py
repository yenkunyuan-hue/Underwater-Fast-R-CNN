import torch
import torch.nn.functional as f
import math
from torchvision.models.detection.roi_heads import RoIHeads
from torchvision.models.detection._utils import BoxCoder
box_coder = BoxCoder(weights=(10.0, 10.0, 5.0, 5.0))

def complete_iou_loss(pred_boxes, target_boxes, eps=1e-7):
    px1, py1, px2, py2 = pred_boxes.unbind(-1)
    tx1, ty1, tx2, ty2 = target_boxes.unbind(-1)

    # intersection
    ix1 = torch.max(px1, tx1); iy1 = torch.max(py1, ty1)
    ix2 = torch.min(px2, tx2); iy2 = torch.min(py2, ty2)
    iw = (ix2 - ix1).clamp(min=0); ih = (iy2 - iy1).clamp(min=0)
    inter = iw * ih

    p_w = (px2 - px1).clamp(min=0); p_h = (py2 - py1).clamp(min=0)
    t_w = (tx2 - tx1).clamp(min=0); t_h = (ty2 - ty1).clamp(min=0)
    union = p_w * p_h + t_w * t_h - inter + eps
    iou = inter / union

    # smallest enclosing box diagonal
    cx1 = torch.min(px1, tx1); cy1 = torch.min(py1, ty1)
    cx2 = torch.max(px2, tx2); cy2 = torch.max(py2, ty2)
    c2 = (cx2 - cx1).pow(2) + (cy2 - cy1).pow(2) + eps

    # normalized center-distance penalty
    p_cx = (px1 + px2) / 2; p_cy = (py1 + py2) / 2
    t_cx = (tx1 + tx2) / 2; t_cy = (ty1 + ty2) / 2
    rho2 = (p_cx - t_cx).pow(2) + (p_cy - t_cy).pow(2)

    # aspect-ratio consistency penalty
    p_w_safe = p_w.clamp(min=eps); p_h_safe = p_h.clamp(min=eps)
    t_w_safe = t_w.clamp(min=eps); t_h_safe = t_h.clamp(min=eps)
    v = (4 / (math.pi ** 2)) * (torch.atan(t_w_safe / t_h_safe) - torch.atan(p_w_safe / p_h_safe)).pow(2)
    with torch.no_grad():
        alpha = v / (1 - iou + v + eps)

    ciou = iou - (rho2 / c2) - alpha * v
    return ciou

class CIoURoIHeads(RoIHeads):
    def forward(self, features, proposals, image_shapes, targets=None):
        if targets is not None:
            for t in targets:
                floating_point_types = (torch.float, torch.double, torch.half)
                if t["boxes"].dtype not in floating_point_types:
                    raise TypeError(f"target boxes must of float type, instead got {t['boxes'].dtype}")
                if t["labels"].dtype != torch.int64:
                    raise TypeError(f"target labels must of int64 type, instead got {t['labels'].dtype}")

        if self.training:
            proposals, matched_idxs, labels, regression_targets = self.select_training_samples(proposals, targets)
        else:
            labels = None
            regression_targets = None
            matched_idxs = None

        box_features = self.box_roi_pool(features, proposals, image_shapes)
        box_features = self.box_head(box_features)
        class_logits, box_regression = self.box_predictor(box_features)

        result = []
        losses = {}
        if self.training:
            if labels is None or regression_targets is None:
                raise ValueError("labels/regression_targets cannot be None in training")

            labels_cat = torch.cat(labels, dim=0)
            classification_loss = torch.nn.functional.cross_entropy(class_logits, labels_cat)

            sampled_pos_inds_subset = torch.where(labels_cat > 0)[0]
            labels_pos = labels_cat[sampled_pos_inds_subset]
            N = class_logits.shape[0]
            box_regression_reshaped = box_regression.reshape(N, box_regression.size(-1) // 4, 4)
            pred_deltas_pos = box_regression_reshaped[sampled_pos_inds_subset, labels_pos]

            proposals_cat = torch.cat(proposals, dim=0)
            proposals_pos = proposals_cat[sampled_pos_inds_subset]

            regression_targets_cat = torch.cat(regression_targets, dim=0)
            target_deltas_pos = regression_targets_cat[sampled_pos_inds_subset]

            # decode both predicted and target deltas back to absolute
            # (x1,y1,x2,y2) coordinates -- CIoU needs real box geometry,
            # not the delta parameterization smooth-L1 operates on.
            pred_boxes_pos = self.box_coder.decode_single(pred_deltas_pos, proposals_pos)
            target_boxes_pos = self.box_coder.decode_single(target_deltas_pos, proposals_pos)

            ciou_vec = complete_iou_loss(pred_boxes_pos, target_boxes_pos)
            ciou_sum = ciou_vec.mean().detach()
            if ciou_sum<=0.2:
                auto_ciou = torch.tensor(0.2, device=ciou_vec.device)
            else:
                auto_ciou = ciou_sum
            b1 = (ciou_vec <= auto_ciou - 0.1).float()
            a1 = 1.0

            b2 = ((ciou_vec > (auto_ciou - 0.1)) & (ciou_vec < auto_ciou)).float()
            a2 = torch.exp(auto_ciou)

            b3 = (ciou_vec >= auto_ciou).float()
            a3 = torch.exp(-(ciou_vec - 1.0))

            # 4. 合成逐样本的动态权重 modulating_weight [M]
            modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
            base_ciou_loss = 1.0 - ciou_vec
            box_loss = (base_ciou_loss * modulating_weight).sum() / max(sampled_pos_inds_subset.numel(),1)

            losses = {"loss_classifier": classification_loss, "loss_box_reg": box_loss}
        else:
            boxes, scores, labels = self.postprocess_detections(class_logits, box_regression, proposals, image_shapes)
            for i in range(len(boxes)):
                result.append({"boxes": boxes[i], "labels": labels[i], "scores": scores[i]})

        return result, losses

class criterion(nn.Module):
    def __init__(self, matcher, weight_dict, class_num=10, back_coef=0.01):
        super().__init__()
        self.matcher = matcher
        self.class_num = class_num
        self.weight_dict = weight_dict
        self.back_coef = back_coef
        initial_weight = torch.ones(class_num)
        initial_weight[0] = back_coef
        self.register_buffer('initial_weight', initial_weight)

    def forward(self, prediction, target):
        num_box = 0
        dice = self.matcher(prediction,target)
        pred_type = prediction['pred_logits']
        pred_box = prediction['pred_boxes']
        box_match_pred = []
        box_match_target = []
        #class_num处object is background
        target_type = torch.full(pred_type.shape[:2],0, device = device)
        for i, (index_i, index_j) in enumerate(dice):
            num_box += len(index_i)
            target_type[i][index_i] = target[i]['labels'][index_j]
            box_match_pred.append(pred_box[i, index_i])
            box_match_target.append(target[i]['boxes'][index_j])
        num_box = max(num_box,1)

        box_match_pred = torch.cat(box_match_pred, 0)
        box_match_target = torch.cat(box_match_target, 0)

        loss_classif = F.cross_entropy(pred_type.transpose(1,2),target_type, reduction='mean')

        loss_bbox = 0
        loss_bbox = F.l1_loss(box_match_pred, box_match_target, reduction='none').sum(dim=-1)

        loss_giou = generalized_box_iou_loss(box_match_target,box_match_pred,reduction='sum')/num_box

        #slid box loss base on ciou
        loss_ciou = complete_iou_loss(box_match_pred, box_match_target).detach()
        loss_ciou_average = loss_ciou.sum()/num_box

        if loss_ciou_average<0.2:
            auto_ciou = torch.tensor(0.2).to(device)
        else:
            auto_ciou = loss_ciou_average

        ciou_vec = loss_ciou
        b1 = (ciou_vec <= auto_ciou - 0.1).float()
        a1 = 1.0

        b2 = ((ciou_vec > (auto_ciou - 0.1)) & (ciou_vec < auto_ciou)).float()
        a2 = torch.exp(auto_ciou)

        b3 = (ciou_vec >= auto_ciou).float()
        a3 = torch.exp(-(ciou_vec - 1.0))

        # 4. 合成逐样本的动态权重 modulating_weight [M]
        modulating_weight = a1 * b1 + a2 * b2 + a3 * b3
        loss_bbox = (loss_bbox * modulating_weight).sum() / num_box
        loss_overall ={
            'loss_giou': loss_giou,
            'loss_classification': loss_classif
        }
        print(loss_overall)
        return loss_overall