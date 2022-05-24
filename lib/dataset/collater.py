import torch
import numpy as np
import random
import math
from datetime import datetime
from collections import defaultdict


class collater():
    def __init__(self, max_patch, mode='random'):
        self.max_patch = max_patch
        self.mode = mode

    def __call__(self, batch_data):
        input_list, pos_mask_list, target_list, target_weight_list, metas = zip(*batch_data)
        if self.max_patch > 0:  # 代表要限制最大patch数
            input_list, pos_mask_list, target_list, target_weight_list, metas = self.get_max_patch(
                list(input_list), list(pos_mask_list), list(target_list), list(target_weight_list), list(metas))
        length = [len(item) for item in input_list]
        input_list = self.concat_batch(input_list)
        pos_mask_list = self.concat_batch(pos_mask_list)
        target_list = self.concat_batch(target_list)
        target_weight_list = self.concat_batch(target_weight_list)
        meta = self.deal_metas(metas, length)

        return input_list, pos_mask_list, target_list, target_weight_list, meta

    def get_max_patch(self, input_list, pos_mask_list, target_list, target_weight_list, metas):
        extend_len = 0
        tmp_metas = []
        input_list_split, pos_mask_list_split, target_list_split, target_weight_list_split, metas_split = [],[],[],[],[]
        for idx, meta in enumerate(metas):
            boxes = meta['box']
            n = len(boxes)
            if self.mode == 'main_target':
                if n > 1:
                    for target_index in range(n):
                        dist = self.get_distances(target_index, boxes)
                        nearby = n if n < self.max_patch else self.max_patch
                        used_index = [item[0] for item in dist][:nearby]
                        input_list_split.extend([self.tailor_list(input_list[idx].copy(), used_index)])
                        pos_mask_list_split.extend([self.tailor_list(pos_mask_list[idx].copy(), used_index)])
                        target_list_split.extend([self.tailor_list(target_list[idx].copy(), used_index)])
                        target_weight_list_split.extend([self.tailor_list(target_weight_list[idx].copy(), used_index)])
                        metas_split.append(self.tailor_metas(metas[idx].copy(), [target_index]))
                else:
                    input_list_split.append(input_list[idx])
                    pos_mask_list_split.append(pos_mask_list[idx])
                    target_list_split.append(target_list[idx])
                    target_weight_list_split.append(target_weight_list[idx])
                    metas_split.append(metas[idx])
            else:
                if n > self.max_patch:
                    if 'random' in self.mode:
                        random.seed(datetime.now())
                        if self.mode == 'random_totally':
                            used_index = random.sample(range(0,n-1), self.max_patch)
                        else:
                            target_index = random.randint(0, n-1)
                            # find the max_patch closest to this person
                            dist = self.get_distances(target_index, boxes)
                            used_index = [item[0] for item in dist][:self.max_patch]
                        input_list[idx] = self.tailor_list(input_list[idx], used_index)
                        pos_mask_list[idx] = self.tailor_list(pos_mask_list[idx], used_index)
                        target_list[idx] = self.tailor_list(target_list[idx], used_index)
                        target_weight_list[idx] = self.tailor_list(target_weight_list[idx], used_index)
                        metas[idx] = self.tailor_metas(metas[idx], used_index)
                    elif self.mode == 'window':
                        # print('当前id: {} | 实际操作id: {} | 数组len={} | extend_len={} '.format(idx, idx+extend_len, len(input_list[idx+extend_len]), extend_len))
                        self.extend_list(input_list, self.max_patch, idx+extend_len)
                        self.extend_list(pos_mask_list, self.max_patch, idx+extend_len)
                        self.extend_list(target_list, self.max_patch, idx+extend_len)
                        self.extend_list(target_weight_list, self.max_patch, idx+extend_len)
                        tmp_metas.append({
                            'idx': idx,
                            'list': self.extend_metas(metas, self.max_patch, idx)
                        })
                        extend_len += (math.ceil(n/self.max_patch) - 1)
                        # print('\t 插入{}份, extend_len={}'.format(math.floor(n/self.max_patch), extend_len))
        
        if self.mode == 'window':
            extend_len = 0
            for item in tmp_metas:
                idx = item['idx']
                item_list = item['list']
                step_idx = idx + extend_len
                metas[step_idx] = item_list[0]
                for i in range(len(item_list)-1):
                    i = i + 1
                    metas.insert(step_idx+i, item_list[i])
                extend_len += (len(item_list) -1)
        if self.mode == 'main_target':
            return input_list_split, pos_mask_list_split, target_list_split, target_weight_list_split, metas_split 
        else:
            return input_list, pos_mask_list, target_list, target_weight_list, metas

    def extend_metas(self, metas, step, step_idx):

        def extend_meta_list(array_list, step):
            tmp_list = [array_list[i:i+step] for i in range(0, len(array_list), step)]
            return tmp_list

        meta = metas[step_idx]

        image = meta['image']
        filename = meta['filename']
        rotation = meta['rotation']
        
        meta['joints'] = extend_meta_list(meta['joints'], step)
        meta['joints_vis'] = extend_meta_list(meta['joints_vis'], step)
        meta['center'] = extend_meta_list(meta['center'], step)
        meta['scale'] = extend_meta_list(meta['scale'], step)

        meta['imgnum'] = extend_meta_list(meta['imgnum'], step)
        meta['score'] = extend_meta_list(meta['score'], step)
        meta['box'] = extend_meta_list(meta['box'], step)

        tmp_list = []
        for i in range(len(meta['joints'])):
            tmp_list.append({
                'image': image,
                'filename': filename,
                'rotation': rotation,
                'imgnum': meta['imgnum'][i],
                'joints': meta['joints'][i],
                'joints_vis': meta['joints_vis'][i],
                'center': meta['center'][i],
                'scale': meta['scale'][i],
                'score': meta['score'][i],
                'box': meta['box'][i]
            })
        
        return tmp_list


    def extend_list(self, array_list, step, step_idx):
        input = array_list[step_idx]
        tmp_list = [input[i:i+step] for i in range(0, len(input), step)]
        array_list[step_idx] = tmp_list[0]
        for i in range(len(tmp_list)-1):
            i = i + 1
            array_list.insert(step_idx+i, tmp_list[i])
    
    
    def tailor_list(self, input, used_index):
        tailor_res = []
        for i in used_index:
            tailor_res.append(input[i])
        return tailor_res

    def tailor_metas(self, meta, used_index):

        meta['joints'] = self.tailor_list(meta['joints'], used_index)
        meta['joints_vis'] = self.tailor_list(meta['joints_vis'], used_index)
        meta['center'] = self.tailor_list(meta['center'], used_index)
        meta['scale'] = self.tailor_list(meta['scale'], used_index)

        meta['imgnum'] = self.tailor_list(meta['imgnum'], used_index)
        meta['score'] = self.tailor_list(meta['score'], used_index)
        meta['box'] = self.tailor_list(meta['box'], used_index)

        return meta

    def get_distances(self, target_index, box):
        target = box[target_index]
        dist = defaultdict(list)
        for i in range(len(box)):
            x = np.array([target[0], target[1]])
            y = np.array([box[i][0], box[i][1]])
            dist[i] = np.linalg.norm(x - y)
        dist = sorted(dist.items(), key=lambda item: item[1])
        # print("dist:", dist)
        return dist

    def concat_batch(self, batchItem):
        res_list = []
        for item in batchItem:
            tmp = torch.stack(item, dim=0)
            res_list.append(tmp)
        res = torch.cat(res_list)
        return res

    def deal_metas(self, metas, length):
        N = max(length)

        def deal_np_array_list(array):
            res_list = []
            for item in array:
                res_list.append(torch.from_numpy(item))
            res = torch.stack(res_list, dim=0)
            return res

        def padding_imgnum(array):
            for _ in range(N-len(array)):
                array.append(0)
            array = np.array(array, dtype=np.int32)
            return torch.from_numpy(array)

        meta = {
            'image': [],
            'length': torch.from_numpy(np.array(length, dtype=np.int32)),
            'filename': [],
            'imgnum': [],
            'rotation': [],
            'joints': [],
            'joints_vis': [],
            'center': [],
            'scale': [],
            'score': [],
            'box': []
        }

        for item in metas:
            meta['image'].append(item['image'])
            meta['filename'].append(item['filename'])
            meta['box'].append(item['box'])
            meta['imgnum'].append(padding_imgnum(item['imgnum']))
            meta['rotation'].append(item['rotation'])
            meta['score'].extend(item['score'])

            meta['joints'].append(deal_np_array_list(item['joints']))
            meta['joints_vis'].append(deal_np_array_list(item['joints_vis']))
            meta['center'].append(deal_np_array_list(item['center']))
            meta['scale'].append(deal_np_array_list(item['scale']))

        meta['joints'] = torch.cat(meta['joints'], dim=0)
        meta['imgnum'] = torch.cat(meta['imgnum'], dim=0)
        meta['joints_vis'] = torch.cat(meta['joints_vis'], dim=0)
        meta['center'] = torch.cat(meta['center'], dim=0)
        meta['scale'] = torch.cat(meta['scale'], dim=0)

        meta['score'] = torch.from_numpy(
            np.array(meta['score'], dtype=np.int32))
        
        # delete unused
        del meta['filename']
        del meta['rotation']
        del meta['box']

        return meta
