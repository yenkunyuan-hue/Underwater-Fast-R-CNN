import os
import matplotlib.pyplot as plt  # FIX: 原来是 import matplotlib as plt，plt.figure 不存在
import cv2

def loss_graph_plot(loss_history):
    # FIX: one_epoch_train 现在只返回总 loss，box loss 那张图已没有数据，只画 train / validation 总 loss
    save_path = '/ceph/home/student.aau.dk/fy14pi/loss_overall'
    # FIX: 原来检查绝对路径、却用相对路径创建目录，换个运行目录就会 FileNotFoundError
    os.makedirs(save_path, exist_ok=True)
    save_path_1 = os.path.join(save_path, 'Average_Loss.png')
    x1 = loss_history[0]
    y1 = loss_history[1]
    y2 = loss_history[2]
    plt.figure()
    plt.plot(x1,y1, label = 'Trainning Loss', color = 'blue')
    plt.plot(x1,y2, label = 'Validation Loss', color = 'red')
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


def graph_plot(path_image, target, prediction, round, test_set):  # FIX: test_set 在本模块未定义，改为参数传入

    object_name = ['cage', 'frame', 'hook', 'anchor', 'tire', 'rov', 'plastic bucket', 'fish', 'oil drum']
    colour_predict_box = (0, 255, 255)  # 亮黄色
    colour_predict_text = (0, 255, 255)  # 亮黄色
    colour_target_box = (255, 0, 255)  # 亮洋红/紫粉色
    colour_target_text = (255, 0, 255)  # 亮洋红/紫粉色

    save_path = '/ceph/home/student.aau.dk/fy14pi/test_picture'
    os.makedirs(save_path, exist_ok=True)  # FIX: 同 loss_graph_plot，原来用相对路径创建目录

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