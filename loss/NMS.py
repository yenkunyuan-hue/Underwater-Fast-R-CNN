import numpy as np
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
