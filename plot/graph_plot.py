import os
import matplotlib as plt
import cv2

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