from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def do_python_keypoint_eval(res_file, gt_file):
    coco_gt = COCO(gt_file)
    coco_dt = coco_gt.loadRes(res_file)

    coco_eval = COCOeval(coco_gt, coco_dt, 'keypoints')
    coco_eval.params.useSegm = None
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats_names = ['AP', 'Ap .5', 'AP .75', 'AP (M)', 'AP (L)', 'AR', 'AR .5', 'AR .75', 'AR (M)', 'AR (L)']

    info_str = []
    for ind, name in enumerate(stats_names):
        info_str.append((name, coco_eval.stats[ind]))

    return info_str


if __name__ == '__main__':
    gt_file = '/media/yiwei/yiwei-01/datasets/pose/coco/annotations/person_keypoints_val2017.json'
    # res_file = '/home/yiwei/Desktop/ori_keypoints_val2017_results_1.json'
    # res_file = '/home/yiwei/Desktop/ori_keypoints_val2017_results_0.json'
    # res_file = '/home/yiwei/Desktop/my_keypoints_val2017_results_0.json'
    # res_file = '/home/yiwei/Desktop/trans_res.json'
    res_file = '/home/yiwei/Desktop/ori_trans_res.json'

    do_python_keypoint_eval(res_file, gt_file)