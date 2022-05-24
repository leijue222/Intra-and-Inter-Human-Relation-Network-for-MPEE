import matplotlib.pyplot as plt
import matplotlib
import math
import numpy as np
import cv2
import os
import json
from tqdm import tqdm

coco_keypoint_name = {
    0: "nose",
    1: "eye(l)",
    2: "eye(r)",
    3: "ear(l)",
    4: "ear(r)",
    5: "sho.(l)",
    6: "sho.(r)",
    7: "elb.(l)",
    8: "elb.(r)",
    9: "wri.(l)",
    10: "wri.(r)",
    11: "hip(l)",
    12: "hip(r)",
    13: "kne.(l)",
    14: "kne.(r)",
    15: "ank.(l)",
    16: "ank.(r)",
}

class plt_config:
    def __init__(self, dataset_name):
        assert dataset_name == "coco", "{} dataset is not supported".format(
            dataset_name
        )
        self.n_kpt = 17
        # edge , color
        self.EDGES = [
            ([15, 13], [255, 0, 0]),  # l_ankle -> l_knee
            ([13, 11], [155, 85, 0]),  # l_knee -> l_hip
            ([11, 5], [155, 85, 0]),  # l_hip -> l_shoulder
            ([12, 14], [0, 0, 255]),  # r_hip -> r_knee
            ([14, 16], [17, 25, 10]),  # r_knee -> r_ankle
            ([12, 6], [0, 0, 255]),  # r_hip  -> r_shoulder
            ([3, 1], [0, 255, 0]),  # l_ear -> l_eye
            ([1, 2], [0, 255, 5]),  # l_eye -> r_eye
            ([1, 0], [0, 255, 170]),  # l_eye -> nose
            ([0, 2], [0, 255, 25]),  # nose -> r_eye
            ([2, 4], [0, 17, 255]),  # r_eye -> r_ear
            ([9, 7], [0, 220, 0]),  # l_wrist -> l_elbow
            ([7, 5], [0, 220, 0]),  # l_elbow -> l_shoulder
            ([5, 6], [125, 125, 155]),  # l_shoulder -> r_shoulder
            ([6, 8], [25, 0, 55]),  # r_shoulder -> r_elbow
            ([8, 10], [25, 0, 255]),
        ]  # r_elbow -> r_wrist


def plt_show_cv2_image(image):
    image0 = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.figure()  # 打开一个画布
    plt.axis('off')  # 不打开坐标轴
    plt.imshow(image0)
    plt.show()  # 加上这个才能显示
    # plt.pause(0.01)  # 暂时显示0.01秒
    # plt.draw()  # 重新绘制当前图形


def plot_poses(
    img, skeletons, config=plt_config("coco"), save_path=None, dataset_name="coco"
):

    cmap = matplotlib.cm.get_cmap("hsv")
    canvas = img.copy()
    n_kpt = config.n_kpt
    for i in range(n_kpt):
        rgba = np.array(cmap(1 - i / n_kpt - 1.0 / n_kpt * 2))
        rgba[0:3] *= 255
        for j in range(len(skeletons)):
            if len(skeletons[j][i]) > 2 and skeletons[j][i, 2] > 0:
                cv2.circle(
                    canvas,
                    tuple(skeletons[j][i, 0:2].astype("int32")),
                    3,
                    (255, 255, 255),
                    thickness=-1,
                )

    stickwidth = 2
    for i in range(len(config.EDGES)):
        for j in range(len(skeletons)):
            edge = config.EDGES[i][0]
            color = config.EDGES[i][1]
            if len(skeletons[j][edge[0]]) > 2:
                if skeletons[j][edge[0], 2] == 0 or skeletons[j][edge[1], 2] == 0:
                    continue

            cur_canvas = canvas.copy()
            X = [skeletons[j][edge[0], 1], skeletons[j][edge[1], 1]]
            Y = [skeletons[j][edge[0], 0], skeletons[j][edge[1], 0]]
            mX = np.mean(X)
            mY = np.mean(Y)
            length = ((X[0] - X[1]) ** 2 + (Y[0] - Y[1]) ** 2) ** 0.5
            angle = math.degrees(math.atan2(X[0] - X[1], Y[0] - Y[1]))
            polygon = cv2.ellipse2Poly(
                (int(mY), int(mX)), (int(length / 2), stickwidth), int(angle), 0, 360, 1
            )
            cv2.fillConvexPoly(cur_canvas, polygon, color)
            canvas = cv2.addWeighted(canvas, 0.4, cur_canvas, 0.6, 0)

    return canvas

json_file='output/OCHuman/interformer_2stage/2stage_transposeH_fixF_lr1_notrans/results/keypoints_ochuman_coco_format_test_range_0.00_1.00.json_results_0.json'
# imgset_path='/mnt/sda2/datasets/OCHuman/images'
imgset_path='/media/yiwei/yiwei-01/datasets/pose/OCHuman/images'
out_path='output/OCHuman/interformer_2stage/2stage_transposeH_fixF_lr1_notrans/vis_output'
with open(json_file,'r',encoding='utf8') as fp:
    json_data = json.load(fp)
    # print(type(json_data[0]))
    # print(len(json_data))
    # print(json_data[0].keys())
    img_id_set = []
    for res in json_data:
        img_id = res['image_id']
        img_id_set.append(img_id)
    for img_id in tqdm(set(img_id_set)):
        kpts=[]
        img_file = os.path.join(imgset_path,'%06d.jpg'%img_id)
        # print(img_file)
        img = cv2.imread(img_file,cv2.COLOR_BGR2RGB)
        for res in json_data:
            if img_id == res['image_id']:
                kpts.append(np.array(res['keypoints']).reshape(17,3))
        
        vis_out = plot_poses(img,kpts)
        # plt_show_cv2_image(vis_out)
        cv2.imwrite(os.path.join(out_path,'%06d.jpg'%img_id),vis_out)
        # break