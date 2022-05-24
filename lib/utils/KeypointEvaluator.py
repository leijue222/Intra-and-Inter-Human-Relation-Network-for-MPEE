import glob
import os
import pickle
import json
import shutil

class KeypointEvaluator() :
    
    def __init__(self) :

        self.__start_points = [1,2,6,10]
        self.__cluter_mode = ClusterMode(self.__start_points) # 默认模式
        
    def __load_index2humans(self, index_file_path) :
        with open(index_file_path, 'rb') as f:
            return pickle.load(f)

    def __read_json(self, path) :
        with open(path, 'r') as f:
            dataset = json.load(f)
        return dataset
    
    def __save_n_humans_json(self, json_obj, n_humans, path) :
        js_obj = json.dumps(json_obj)

        if not os.path.exists(path):
            os.mkdir(path)

        fileObject = open(
            os.path.join(path, 'keypoints_n_humans_')+str(n_humans)+'.json', 'w')

        fileObject.write(js_obj)
        fileObject.close()

        print(os.path.join(path, 'keypoints_n_humans_')+str(n_humans)+'.json   '+'保存成功')

    def __save_n_humans_count_json(self, n_humans_list, n_humans_gt_images, path) :

        if not os.path.exists(path):
            os.mkdir(path)

        n_humans_count = {}

        for i in n_humans_list:
            n_humans_count[i] = len(n_humans_gt_images[i])
        
        fileObject = open(os.path.join(path, 'keypoints_n_humans_count')+'.json', 'w')
        
        js_obj = json.dumps(n_humans_count)
        fileObject.write(js_obj)
        fileObject.close()

        print(os.path.join(path, 'keypoints_n_humans_count')+'.json    '+'保存成功')
    
    def __create_coco_subdataset(self, info, tg_categories, tg_annotations, tg_images, tg_licenses) :
        subdataset = dict()
        subdataset['info'] = info
        subdataset['licenses'] = tg_licenses
        subdataset['images'] = tg_images
        subdataset['annotations'] = tg_annotations
        subdataset['categories'] = tg_categories
        return subdataset

    def __create_crowdpose_subdataset(self, info, tg_categories, tg_annotations, tg_images) :
        subdataset = dict()
        subdataset['info'] = info
        subdataset['images'] = tg_images
        subdataset['annotations'] = tg_annotations
        subdataset['categories'] = tg_categories
        return subdataset

    def __create_OCHuman_subdataset(self, tg_categories, tg_annotations, tg_images) :
        subdataset = dict()
        subdataset['images'] = tg_images
        subdataset['annotations'] = tg_annotations
        subdataset['categories'] = tg_categories
        return subdataset

    def __get_n_humans_coco_gt_json(self, index_file_path, gt_file_path,  is_save = False) :
        gt_file = self.__read_json(gt_file_path)
        
        info = gt_file['info']
        categories = gt_file['categories']
        annotations = gt_file['annotations']
        images = gt_file['images']
        licenses = gt_file['licenses']

        index2humans = self.__load_index2humans(index_file_path)

        n_humans_gt_images={}
        n_humans_gt_images_ids={}
        n_humans_gt_anns={}
        n_humans_list=[]

        for image in images:
            n_humans=index2humans[str(image['id']).zfill(6)] # 有几个人
            if n_humans != 0:
                if n_humans not in n_humans_gt_images.keys():
                    n_humans_gt_images[n_humans]=[]
                n_humans_gt_images[n_humans].append(image)
                if n_humans not in n_humans_gt_images_ids.keys():
                    n_humans_gt_images_ids[n_humans]=[]
                n_humans_gt_images_ids[n_humans].append(image['id'])
                if n_humans not in n_humans_list:
                    n_humans_list.append(n_humans)
        
        for ann in annotations:
            for i in n_humans_list:
                if ann['image_id'] in n_humans_gt_images_ids[i]:
                    if i not in n_humans_gt_anns.keys():
                        n_humans_gt_anns[i]=[]
                    n_humans_gt_anns[i].append(ann)

        n_humans_coco_gt_file = {}

        for i in n_humans_list:
            subdataset = self.__create_coco_subdataset(info,categories,n_humans_gt_anns[i],n_humans_gt_images[i],licenses)
            n_humans_coco_gt_file[i] = subdataset

        gt_n_humans_dir = 'gt_n_humans'
        if is_save:
            for i in n_humans_list:
                self.__save_n_humans_json(n_humans_coco_gt_file[i],i, os.path.join(os.path.dirname(gt_file_path), gt_n_humans_dir))

        self.__save_n_humans_count_json(n_humans_list, n_humans_gt_images, os.path.join(os.path.dirname(gt_file_path), gt_n_humans_dir))

        return n_humans_list, n_humans_coco_gt_file

    def __get_n_humans_cluster_coco_gt_json(self, index_file_path, gt_file_path, is_save = True) :

        n_humans_list, n_humans_coco_gt_file = self.__get_n_humans_coco_gt_json(index_file_path, gt_file_path)
        
        cluster_level_list = []

        """这三个所有的都一样"""
        info = n_humans_coco_gt_file[n_humans_list[0]]['info']
        categories = n_humans_coco_gt_file[n_humans_list[0]]['categories']
        licenses = n_humans_coco_gt_file[n_humans_list[0]]['licenses']
        
        n_humans_cluster_coco_images = {}
        n_humans_cluster_coco_anns = {}

        for i in n_humans_list:
            cluster_level = self.__cluter_mode.get_cluster_level(i)
            if cluster_level not in n_humans_cluster_coco_images.keys():    
                n_humans_cluster_coco_images[cluster_level] = []
                n_humans_cluster_coco_anns[cluster_level] = [] 
            
            n_humans_cluster_coco_images[cluster_level].extend(n_humans_coco_gt_file[i]['images'])
            n_humans_cluster_coco_anns[cluster_level].extend(n_humans_coco_gt_file[i]['annotations'])


            if cluster_level not in cluster_level_list:
                cluster_level_list.append(cluster_level)

        n_humans_cluster_coco_gt_file = {}

        for i in cluster_level_list:
            print(i+':'+str(len(n_humans_cluster_coco_images[i])))
            subdataset = self.__create_coco_subdataset(info, categories, n_humans_cluster_coco_anns[i], n_humans_cluster_coco_images[i],licenses)
            n_humans_cluster_coco_gt_file[i] = subdataset

        gt_n_humans_dir = 'gt_n_humans'
        if is_save:
            for i in cluster_level_list:
                self.__save_n_humans_json(n_humans_cluster_coco_gt_file[i], i, os.path.join(os.path.dirname(gt_file_path), gt_n_humans_dir))

    def __get_n_humans_crowdpose_gt_json(self, index_file_path, gt_file_path,  is_save = False) :
        gt_file = self.__read_json(gt_file_path)
            
        info = gt_file['info']
        categories = gt_file['categories']
        annotations = gt_file['annotations']
        images = gt_file['images']

        index2humans = self.__load_index2humans(index_file_path)
        
        n_humans_gt_images = {}
        n_humans_gt_images_ids = {}
        n_humans_gt_anns = {}
        n_humans_list = []

        for image in images:
            n_humans=index2humans[str(image['id'])] # 有几个人
            if n_humans not in n_humans_gt_images.keys():
                n_humans_gt_images[n_humans]=[]
            n_humans_gt_images[n_humans].append(image)
            if n_humans not in n_humans_gt_images_ids.keys():
                n_humans_gt_images_ids[n_humans]=[]
            n_humans_gt_images_ids[n_humans].append(image['id'])
            if n_humans not in n_humans_list:
                n_humans_list.append(n_humans)
        
        for ann in annotations:
            for i in n_humans_list:
                if ann['image_id'] in n_humans_gt_images_ids[i]:
                    if i not in n_humans_gt_anns.keys():
                        n_humans_gt_anns[i]=[]
                    n_humans_gt_anns[i].append(ann)

        n_humans_crowdpose_gt_file = {}

        for i in n_humans_list:
            subdataset = self.__create_crowdpose_subdataset(info,categories,n_humans_gt_anns[i],n_humans_gt_images[i])
            n_humans_crowdpose_gt_file[i] = subdataset

        gt_n_humans_dir = 'gt_n_humans'
        if is_save:
            for i in n_humans_list:
                self.__save_n_humans_json(n_humans_crowdpose_gt_file[i], i, os.path.join(os.path.dirname(gt_file_path),gt_n_humans_dir))

        self.__save_n_humans_count_json(n_humans_list, n_humans_gt_images, os.path.join(os.path.dirname(gt_file_path),gt_n_humans_dir))

        return n_humans_list, n_humans_crowdpose_gt_file
   
    def __get_n_humans_cluster_crowdpose_gt_json(self, index_file_path, gt_file_path, is_save = True) :
        
        n_humans_list, n_humans_crowdpose_gt_file = self.__get_n_humans_crowdpose_gt_json(index_file_path, gt_file_path)
        
        cluster_level_list = []

        """这三个所有的都一样"""
        info = n_humans_crowdpose_gt_file[n_humans_list[0]]['info']
        categories = n_humans_crowdpose_gt_file[n_humans_list[0]]['categories']
        
        n_humans_cluster_crowdpose_images = {}
        n_humans_cluster_crowdpose_anns = {}

        for i in n_humans_list:
            cluster_level = self.__cluter_mode.get_cluster_level(i)
            if cluster_level not in n_humans_cluster_crowdpose_images.keys():    
                n_humans_cluster_crowdpose_images[cluster_level] = []
                n_humans_cluster_crowdpose_anns[cluster_level] = [] 
            
            n_humans_cluster_crowdpose_images[cluster_level].extend(n_humans_crowdpose_gt_file[i]['images'])
            n_humans_cluster_crowdpose_anns[cluster_level].extend(n_humans_crowdpose_gt_file[i]['annotations'])


            if cluster_level not in cluster_level_list:
                cluster_level_list.append(cluster_level)

        n_humans_cluster_crowdpose_gt_file = {}

        for i in cluster_level_list:
            print(i+':'+str(len(n_humans_cluster_crowdpose_images[i])))
            subdataset = self.__create_crowdpose_subdataset(info, categories, n_humans_cluster_crowdpose_anns[i], n_humans_cluster_crowdpose_images[i])
            n_humans_cluster_crowdpose_gt_file[i] = subdataset

        if is_save:
            for i in cluster_level_list:
                gt_n_humans_dir = 'gt_n_humans'
                self.__save_n_humans_json(n_humans_cluster_crowdpose_gt_file[i], i, os.path.join(os.path.dirname(gt_file_path),gt_n_humans_dir))
    
    def __get_n_humans_OCHuman_gt_json(self, index_file_path, gt_file_path,  is_save = False) :
        
        gt_file = self.__read_json(gt_file_path)
        
        categories = gt_file['categories']
        annotations = gt_file['annotations']
        images = gt_file['images']

        index2humans = self.__load_index2humans(index_file_path)

        n_humans_gt_images={}
        n_humans_gt_images_ids={}
        n_humans_gt_anns={}
        n_humans_list=[]

        for image in images:
            n_humans=index2humans[str(image['id']).zfill(6)] # 有几个人
            if n_humans != 0:
                if n_humans not in n_humans_gt_images.keys():
                    n_humans_gt_images[n_humans]=[]
                n_humans_gt_images[n_humans].append(image)
                if n_humans not in n_humans_gt_images_ids.keys():
                    n_humans_gt_images_ids[n_humans]=[]
                n_humans_gt_images_ids[n_humans].append(image['id'])
                if n_humans not in n_humans_list:
                    n_humans_list.append(n_humans)
        
        for ann in annotations:
            for i in n_humans_list:
                if ann['image_id'] in n_humans_gt_images_ids[i]:
                    if i not in n_humans_gt_anns.keys():
                        n_humans_gt_anns[i]=[]
                    n_humans_gt_anns[i].append(ann)

        n_humans_OCHuman_gt_file = {}

        for i in n_humans_list:
            subdataset = self.__create_OCHuman_subdataset(categories,n_humans_gt_anns[i],n_humans_gt_images[i])
            n_humans_OCHuman_gt_file[i] = subdataset

        gt_n_humans_dir = 'gt_n_humans'
        if is_save:
            for i in n_humans_list:
                self.__save_n_humans_json(n_humans_OCHuman_gt_file[i],i, os.path.join(os.path.dirname(gt_file_path), gt_n_humans_dir))

        self.__save_n_humans_count_json(n_humans_list, n_humans_gt_images, os.path.join(os.path.dirname(gt_file_path), gt_n_humans_dir))

        return n_humans_list, n_humans_OCHuman_gt_file

    def __get_n_humans_cluster_OCHuman_gt_json(self, index_file_path, gt_file_path, is_save = True) : 
        n_humans_list, n_humans_OCHuman_gt_file = self.__get_n_humans_OCHuman_gt_json(index_file_path, gt_file_path)
        
        cluster_level_list = []

        """这个所有的都一样"""
        categories = n_humans_OCHuman_gt_file[n_humans_list[0]]['categories']
        
        n_humans_cluster_crowdpose_images = {}
        n_humans_cluster_crowdpose_anns = {}

        for i in n_humans_list:
            cluster_level = self.__cluter_mode.get_cluster_level(i)
            if cluster_level not in n_humans_cluster_crowdpose_images.keys():    
                n_humans_cluster_crowdpose_images[cluster_level] = []
                n_humans_cluster_crowdpose_anns[cluster_level] = [] 
            
            n_humans_cluster_crowdpose_images[cluster_level].extend(n_humans_OCHuman_gt_file[i]['images'])
            n_humans_cluster_crowdpose_anns[cluster_level].extend(n_humans_OCHuman_gt_file[i]['annotations'])


            if cluster_level not in cluster_level_list:
                cluster_level_list.append(cluster_level)

        n_humans_cluster_crowdpose_gt_file = {}

        for i in cluster_level_list:
            print(i+':'+str(len(n_humans_cluster_crowdpose_images[i])))
            subdataset = self.__create_OCHuman_subdataset(categories, n_humans_cluster_crowdpose_anns[i], n_humans_cluster_crowdpose_images[i])
            n_humans_cluster_crowdpose_gt_file[i] = subdataset

        if is_save:
            for i in cluster_level_list:
                gt_n_humans_dir = 'gt_n_humans'
                self.__save_n_humans_json(n_humans_cluster_crowdpose_gt_file[i], i, os.path.join(os.path.dirname(gt_file_path), gt_n_humans_dir))

    def __get_n_humans_res_json(self, index_file_path, res_file_path,  is_save = False) :
        res_file = self.__read_json(res_file_path)

        index2humans = self.__load_index2humans(index_file_path)
        n_humans_res_file={}
        n_humans_list=[]

        for res in res_file:
            # n_humans=index2humans[res['image_id']] # coco
            n_humans=index2humans[str(res['image_id']).zfill(6)] # 有几个人 crowdpose
            if n_humans != 0:
                if n_humans not in n_humans_res_file.keys():
                    n_humans_res_file[n_humans]=[]
                if n_humans not in n_humans_list:
                    n_humans_list.append(n_humans)
                n_humans_res_file[n_humans].append(res)
        if is_save:
            res_n_humans_dir = 'res_n_humans'
            for i in n_humans_list:
                self.__save_n_humans_json(n_humans_res_file[i], i, os.path.join(os.path.dirname(res_file_path),res_n_humans_dir))
            
        return n_humans_list, n_humans_res_file
    
    def __get_n_humans_cluster_res_json(self, index_file_path, res_file_path, is_save = True) :
        n_humans_cluster_res_file = {}
        cluster_level_list = []

        n_humans_list, n_humans_res_file = self.__get_n_humans_res_json(index_file_path, res_file_path)

        for i in n_humans_list:
            cluster_level = self.__cluter_mode.get_cluster_level(i)
            if cluster_level not in n_humans_cluster_res_file.keys():
                n_humans_cluster_res_file[cluster_level] = []
            n_humans_cluster_res_file[cluster_level].extend(n_humans_res_file[i])
            
            if cluster_level not in cluster_level_list:
                cluster_level_list.append(cluster_level)
        
        for i in cluster_level_list:
            print(i+':'+str(len(n_humans_cluster_res_file[i])))

        if is_save:
            res_n_humans_dir = 'res_n_humans'
            for i in cluster_level_list:
                self.__save_n_humans_json(n_humans_cluster_res_file[i], i, os.path.join(os.path.dirname(res_file_path), res_n_humans_dir))
        
        return n_humans_list, n_humans_cluster_res_file

    def __do_coco_python_keypoint_eval(self, res_file, gt_file) :
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
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
            info_str.append((name, coco_eval.stats[ind], round(coco_eval.stats[ind], 3)))

        return info_str

    def __do_crowdpose_python_keypoint_eval(self, res_file, gt_file) :
        from crowdposetools.coco import COCO
        from crowdposetools.cocoeval import COCOeval
        cocoGt = COCO(gt_file)
        cocoDt = cocoGt.loadRes(res_file)
        cocoEval = COCOeval(cocoGt, cocoDt, 'keypoints')
        cocoEval.evaluate()
        cocoEval.accumulate()
        cocoEval.summarize()

        stats_val = cocoEval.stats.tolist()
        stats_val = stats_val[0:3] + stats_val[5:]
        stats_names = ['AP', 'Ap .5', 'AP .75', 'AR', 'AR .5', 'AR .75', 'AP (easy)', 'AP (medium)', 'AP (hard)']

        info_str = []
        for name, val in zip(stats_names, stats_val):
            info_str.append((name, val, round(val, 3)))

        return info_str

    def __write_eval_info(self, gt_file_path, res_file_path, gt_n_humans_files_dir, res_n_humans_files_dir, do_keypoint_eval) :
        gt_n_humans_count = self.__read_json(os.path.join(gt_n_humans_files_dir, 'keypoints_n_humans_count.json'))
        
        gt_n_humans_files_path_list = glob.glob(os.path.join(gt_n_humans_files_dir, 'keypoints_n_humans_c*'))
        res_n_humans_files_path_list = glob.glob(os.path.join(res_n_humans_files_dir, 'keypoints_n_humans_c*'))
        
        all_info_str = do_keypoint_eval(res_file_path, gt_file_path)

        with open(os.path.join(os.path.dirname(res_n_humans_files_dir), 'res_eval.txt'), 'w') as f :

            def humans_count2str(gt_n_humans_count) :
                all_count = {}

                levels = list(set(self.__cluter_mode.cluster.values()))
                levels.append(self.__cluter_mode.get_cluster_level(max(self.__cluter_mode.cluster.keys())+1))

                for i in levels :
                    all_count[i] = {}
                    all_count[i]['total'] = 0

                for i in sorted(gt_n_humans_count.keys()) :
                    all_count[self.__cluter_mode.get_cluster_level(int(i))]['total'] += gt_n_humans_count[i]
                    all_count[self.__cluter_mode.get_cluster_level(int(i))][str(i)] = gt_n_humans_count[i]
                return all_count
            
            all_count = humans_count2str(gt_n_humans_count)
            f.write('\n\n') 
            f.write('All eval:')
            f.write('\n')
            for s in all_info_str:
                f.write(str(s))
                f.write('\n')
            f.write('\n\n') 

            def write_cluster_info(f, count_dict):
                f.write('\n')
                f.write('{'+'\n')
                for key in count_dict.keys():
                    f.write('    '+key+':'+str(count_dict[key]))
                    f.write('\n')
                f.write('}'+'\n')

            for res,gt in zip(res_n_humans_files_path_list, gt_n_humans_files_path_list):
                print('Class {} eval:'.format(res.split("_")[-1]))
                
                f.write('Class {} eval:'.format(res.split("_")[-1]))
                write_cluster_info(f, all_count[res.split("_")[-1].replace('.json','')])
                f.write('\n')

                info_str = do_keypoint_eval(res, gt)
                for s in info_str:
                    f.write(str(s))
                    f.write('\n')
                f.write('\n\n')

    def eval(self, dataset, index_file_path, gt_file_path, res_file_path, cluster_mode = [1,2,6,10]):
        
        gt_parent_path = os.path.dirname(gt_file_path) # 父目录
        res_parent_path = os.path.dirname(res_file_path)
        gt_n_humans_dir = 'gt_n_humans'
        res_n_humans_dir = 'res_n_humans'
        
        if (len(cluster_mode)) > 0 :
            if os.path.exists(os.path.join(gt_parent_path, gt_n_humans_dir)):
                shutil.rmtree(os.path.join(gt_parent_path, gt_n_humans_dir))
            
            if os.path.exists(os.path.join(res_parent_path, res_n_humans_dir)) :
                shutil.rmtree(os.path.join(res_parent_path, res_n_humans_dir))

            self.__change_cluster_mode(cluster_mode)

        if not os.path.exists(os.path.join(gt_parent_path, gt_n_humans_dir)):

            if dataset == 'coco' :
                self.__get_n_humans_cluster_coco_gt_json(index_file_path, gt_file_path)
            elif dataset == 'crowdpose' :
                self.__get_n_humans_cluster_crowdpose_gt_json(index_file_path, gt_file_path)
            elif dataset == 'OCHuman' :
                self.__get_n_humans_cluster_OCHuman_gt_json(index_file_path, gt_file_path)


        self.__get_n_humans_cluster_res_json(index_file_path, res_file_path)

        if dataset == 'coco' or dataset == 'OCHuman':
            self.__write_eval_info(gt_file_path,
                res_file_path,  
                os.path.join(gt_parent_path, gt_n_humans_dir),
                os.path.join(res_parent_path, res_n_humans_dir),
                self.__do_coco_python_keypoint_eval)

        elif dataset == 'crowdpose' :
            self.__write_eval_info(gt_file_path,
                res_file_path,  
                os.path.join(gt_parent_path, gt_n_humans_dir),
                os.path.join(res_parent_path, res_n_humans_dir),
                self.__do_crowdpose_python_keypoint_eval)

    def __change_cluster_mode(self, start_points = [1,2,6,10]) :
        self.__start_points = start_points
        self.__cluter_mode = ClusterMode(start_points)

class ClusterMode():

    def __init__(self, start_points) :
        self.cluster = {}

        for i,_ in enumerate(start_points) :
            if i+1 < len(start_points) :
                for j in range(start_points[i], start_points[i+1]) :
                    self.cluster[j] = 'c'+str(i+1)


    def get_cluster_level(self, n_humans) :
        if n_humans in self.cluster.keys() :
            return self.cluster[n_humans]
        if n_humans>max(self.cluster.keys()) :
            level = self.cluster[max(self.cluster.keys())]
            return level[0]+str(int(level.split('c')[1])+1)