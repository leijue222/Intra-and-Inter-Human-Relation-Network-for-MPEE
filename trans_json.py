from logging import NOTSET
import os
import json


def find_res(file, image_id, center):
    res = None
    for i, item in enumerate(file):
        id = item['image_id']
        my_center = item['center']
        if id == image_id and my_center == center:
            res = item.copy()
            del file[i]
            return res

if __name__ == '__main__':
    ori_file = '/home/yiwei/Desktop/ori_keypoints_val2017_results_0.json'
    my_file = '/home/yiwei/Desktop/my_keypoints_val2017_results_0.json'
    trans_res = []

    with open(ori_file) as f:
        ori_list = json.load(f)
    with open(my_file) as f:
        my_list = json.load(f)

    oris = ori_list.copy()
    mys = my_list.copy()

    for my in mys:
        image_id = my['image_id']
        center = my['center']
        res = find_res(oris, image_id, center)
        if res:
            trans_res.append(res)

    print('ori_list = {}\nmy_list = {}\ntrans_res = {} \n'.format(len(ori_list), len(my_list), len(trans_res)))

    with open("./ori_trans_res.json", 'w') as f:
        json.dump(trans_res, f, sort_keys=True, indent=4)

    print('Done!')
