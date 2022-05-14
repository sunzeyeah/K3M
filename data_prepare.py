# -*- coding: utf-8 -*-

import os
import sys
import traceback
import logging
import argparse
import json
import pandas as pd
import cv2
import random
import csv
import torch
import glob
import jieba
import numpy as np
import tensorpack.dataflow as td

from tqdm import tqdm
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.modeling.postprocessing import detector_postprocess
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers, FastRCNNOutputs, \
    fast_rcnn_inference_single_image


logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)s]  %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


NUM_OBJECTS = 36

FIELDNAMES = ['image_id', 'image_w', 'image_h', 'num_boxes', 'boxes', 'features', 'cls_prob']

MIN_BOXES = 36
MAX_BOXES = 36

FILE_SYSTEM_SEP = "\\" if sys.platform.startswith("win") else "/"


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True,  type=str, help="原始数据文件地址")
    parser.add_argument("--output_dir", required=True,  type=str, help="处理完成后数据文件地址")
    parser.add_argument("--file_item_info", required=True,  type=str, help="文件名-商品信息")
    # parser.add_argument("--file_pair", required=True,  type=str, help="文件名-是否同款商品")
    parser.add_argument("--file_image", required=True,  type=str, help="文件名-商品图片")

    # Optional parameters
    parser.add_argument("--cv_model_config", default="D:\\Data\\ccks2022\\task9\\output\\cv_model\\VG-Detection\\faster_rcnn_R_101_C4_caffe.yaml",  type=str, help="图片特征抽取模型的配置文件")
    parser.add_argument("--cv_model_file", default="D:\\Data\\ccks2022\\task9\\output\\cv_model\\faster_rcnn_from_caffe.pkl",  type=str, help="图片特征抽取模型文件")
    parser.add_argument("--is_cuda", action="store_true", help="是否使用gpu")

    return parser.parse_args()


def rename_images(args, dtype):
    ''' 图谱数据处理

    原图片文件名：url.jpg

    新图片文件名：{item_id}_{dtype}.jpg

    :param args:
    :param dtype: 数据类型， 取值范围: "train", "valid"

    :return:
    '''
    file_item_info = os.path.join(args.data_dir, args.file_item_info)
    # in_image = os.path.join(args.data_dir, args.file_image.format(dtype))
    in_image = os.path.join(args.data_dir, f"item_{dtype}_images_small")
    # out_file = os.path.join(args.output_dir, f'id_title_pvs_cls.txt.{dtype}')
    out_image = os.path.join(args.output_dir, args.file_image.format(dtype))

    if not os.path.exists(out_image):
        os.mkdir(out_image)

    # with open(out_file, 'w', encoding="utf-8") as f_out:
    #     f_out.write("\t".join(("item_id", "cate_name", "title", "pic_name_full", "item_pvs")) + "\n")
    with open(file_item_info, 'r', encoding='UTF-8', errors='ignore') as f_in:
        # item_count = 0
        for line in tqdm(f_in):
            try:
                jd = json.loads(line.strip())
                item_id = jd['item_id']
                item_image_name = jd['item_image_name']
                # title = jd['title']
                # item_pvs = jd['item_pvs']
                # cate_name = jd['cate_name']

                pic_type = item_image_name.split('.')[-1]
                # pic_name = f"{item_count}_{dtype}"
                pic_name_full = f"{item_id}_{dtype}.{pic_type}"
                image_path = os.path.join(in_image, item_image_name)
                image_path_new = os.path.join(out_image, pic_name_full)
                os.system(f"copy {image_path} {image_path_new}")
            except Exception as e:  #
                logger.error(f"[Error] rename images, item id: {item_id}", e)
                # traceback.print_exc()
                continue

            # f_out.write('\t'.join([item_id, cate_name, title, pic_name_full, item_pvs]) + '\n')
            # f_out.flush()
            #
            # item_count += 1


def write_json(file, data):
    f=open(file,"w",encoding="utf-8")
    json.dump(data,f,indent=2,ensure_ascii=False)
    return


def load_image_ids(raw_file_path):
    id_ids = []
    id_titles =[]
    id_picnames = []
    id_pvs =[]
    id_itemIDs =[]
    id_categorys = []
    with open(raw_file_path, 'r', encoding="utf-8") as f:
        for line in tqdm(f):

            image_id, title, pic_name, pvs, category, item_ID = line.strip().split('\t') # 最后是原item_id
            id_ids.append(image_id)

            id_titles.append(title)
            id_picnames.append(pic_name)
            id_pvs.append(pvs)
            id_itemIDs.append(item_ID)
            id_categorys.append(category)

    df = pd.DataFrame({'image_id':id_ids,'caption':id_titles,'pic':id_picnames,'pv':id_pvs,'itemID':id_itemIDs,'category':id_categorys})
    return df


def generate_lmdb_df(args, dtype): ##json文件产自所有文件，而提取图片特征文件不需要
    in_file = os.path.join(args.output_dir, f'id_title_pvs_cls.txt.{dtype}')
    # in_files = [os.path.join(args.output_dir, f'id_title_pvs_cls.txt.{dtype}')]
    out_dir = os.path.join(args.output_dir, "lmbd")
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)

    # if dtype == 'valid':
    #     all_raw_file_path=['./data/id_title_pvs_cls.txt1'] # 1号是valid
    # else:
    #     file_num = len(glob.glob('./data/id_title_pvs_cls.txt*'))
    #     print('file_num:',file_num) #0 (2 3 4 5 6 7 8 9 ..)
    #     all_raw_file_path = ['./data/id_title_pvs_cls.txt'+str(i) for i in range(file_num)]
    #     all_raw_file_path.remove('./data/id_title_pvs_cls.txt1')
    # print(all_raw_file_path)

    # dfs = []
    # for index, infile in enumerate(in_files):
    #     # print(index, ':', infile)
    #     dfs.append(load_image_ids(infile))
    #
    #     # if index == 0:
    #     #     all_df = this_df
    #     # else:
    #     #     all_df = pd.concat([all_df, this_df])
    # df = pd.concat(dfs)

    df = pd.read_csv(in_file, sep="\t")
    df.to_csv(os.path.join(out_dir, f"df_{dtype}.tsv"),
              sep="\t", encoding="utf-8", index=False)
    # print(all_df)


def generate_lmdb_json(train_or_val):
    all_df=pd.read_csv("./data/image_lmdb_json/df_"+train_or_val+".csv",encoding = "utf-8")
    print(all_df)
    for this_obj in ['caption','pic','pv','itemID','category']:
        this_json = []
        for image_id, value in zip(all_df['image_id'], all_df[this_obj]):
            this_json.append((image_id,value))
        write_json("./data/image_lmdb_json/"+this_obj+"_"+train_or_val+".json",this_json)


def get_detections_from_image(predictor, raw_image):
    with torch.no_grad():
        raw_height, raw_width = raw_image.shape[:2]
        # logger.debug(f"original image size: {raw_height}*{raw_width}")

        # Preprocessing
        image = predictor.transform_gen.get_transform(raw_image).apply_image(raw_image)
        # logger.debug(f"Transformed image size: {image.shape[:2]}")
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        inputs = [{"image": image, "height": raw_height, "width": raw_width}]
        images = predictor.model.preprocess_image(inputs)

        # Run Backbone Res1-Res4
        features = predictor.model.backbone(images.tensor)

        # Generate proposals with RPN
        proposals, _ = predictor.model.proposal_generator(images, features, None)
        proposal = proposals[0]
        # logger.debug(f'Proposal Boxes size: {proposal.proposal_boxes.tensor.shape}') # 154 x 4

        # Run RoI head for each proposal (RoI Pooling + Res5)
        proposal_boxes = [x.proposal_boxes for x in proposals]
        features = [features[f] for f in predictor.model.roi_heads.in_features]
        box_features = predictor.model.roi_heads._shared_roi_transform(
            features, proposal_boxes
        )
        feature_pooled = box_features.mean(dim=[2, 3])  # pooled to 1x1
        # logger.debug(f'Pooled features size: {feature_pooled.shape}') # 154 x 2048

        # Predict classes and boxes for each proposal.
        pred_class_logits, pred_proposal_deltas = predictor.model.roi_heads.box_predictor(feature_pooled)
        outputs = FastRCNNOutputs(
            predictor.model.roi_heads.box2box_transform,
            pred_class_logits,
            pred_proposal_deltas,
            proposals,
            predictor.model.roi_heads.smooth_l1_beta,
        )
        # logger.debug(f"FastRCNNOutputs outputs: {outputs}")
        probs = outputs.predict_probs()[0]
        boxes = outputs.predict_boxes()[0]

        # NMS
        for nms_thresh in np.arange(0.5, 1.0, 0.1):
            instances, ids = fast_rcnn_inference_single_image(
                boxes, probs, image.shape[1:],
                score_thresh=0.2, nms_thresh=nms_thresh, topk_per_image=NUM_OBJECTS
            )
            if len(ids) == NUM_OBJECTS:
                break
        # 由800 x 800 换成 736 x 736
        instances = detector_postprocess(instances, raw_height, raw_width) # 恢复出原来的图片大小 # 36 * 4
        # logger.debug(f"instances: {instances.pred_boxes.tensor.cpu().numpy()}")
        # roi_features: 36 x 2048
        roi_features = feature_pooled[ids].detach()
        # logger.debug(f"roi_features size: {roi_features.size()}")
        # selected_probs: 36 * 1601
        selected_probs = probs[ids]
        # logger.debug(f"selected_probs size: {selected_probs.size()}")

        if torch.sum(torch.isnan(roi_features))>0:
            return

        return {
            # "image_id": image_id,
            "image_h": raw_height,
            "image_w": raw_width,
            "num_boxes": len(ids),
            "boxes": instances.pred_boxes.tensor.cpu().numpy(),
            "features": roi_features.cpu().numpy(),
            "cls_prob": selected_probs.cpu().numpy()
            # "boxes": base64.b64encode(instances.pred_boxes.tensor.cpu().numpy()),
            # "features": base64.b64encode(roi_features.cpu().numpy()),
            # "cls_prob": base64.b64encode(selected_probs.cpu().numpy())
        }


def get_predictor(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.cv_model_config)
    # cfg.merge_from_file("./py-bottom-up-attention/configs/VG-Detection/faster_rcnn_R_101_C4_caffe.yaml")
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 300
    cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST = 0.6
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.2
    # VG Weight
    cfg.MODEL.WEIGHTS = args.cv_model_file
    # cfg.MODEL.WEIGHTS = "./faster-rcnn-pkl/faster_rcnn_from_caffe.pkl"
    # Device
    cfg.MODEL.DEVICE = "cuda" if args.is_cuda else "cpu"

    predictor = DefaultPredictor(cfg)
    logger.info(f"predictor: {predictor}")

    return predictor


def generate_image_features(args, dtype):
    out_file = os.path.join(args.output_dir, f"image_features_{dtype}.tsv")
    out_image_pattern = os.path.join(args.output_dir, f"item_{dtype}_images", "*")
    out_image_paths = glob.glob(out_image_pattern)
    logger.info(f"[Step 3] Starting {dtype} image feature extraction, # images : {len(out_image_paths)}")

    # get predictor
    predictor = get_predictor(args)

    # generate image features and save
    tsvfile = open(out_file, "w", encoding="utf-8")
    writer = csv.DictWriter(tsvfile, delimiter="\t", fieldnames=FIELDNAMES)

    for image_path in tqdm(out_image_paths):
        image = cv2.imread(image_path)
        image_id = image_path.split(FILE_SYSTEM_SEP)[-1].split(".")[0]
        try:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            detection_feature = get_detections_from_image(predictor, image_rgb, image_id)
            if detection_feature is None:
                continue
            writer.writerow(detection_feature)
        except cv2.error as e:
            logger.error(f"[CV2 ERROR] image_id: {image_id}", e)
            # traceback.print_exc()
        except Exception as e:
            logger.error(f"[ERROR] image_id: {image_id}", e)
            # traceback.print_exc()


class Conceptual_Caption(td.RNGDataFlow):
    """
    """
    def __init__(self, args, file_type, predictor, shuffle=True):
        """
        Same as in :class:`ILSVRC12`.
        """
        self.args = args
        self.file_type = file_type
        self.image_dir = os.path.join(args.data_dir, args.file_image.format(file_type))
        file_item_info = os.path.join(args.data_dir, args.file_item_info.format(file_type))
        self.lines = []
        ct = 0
        with open(file_item_info, "r", encoding="utf-8") as r:
            while True:
                line = r.readline()
                if not line:
                    break
                jd = json.loads(line.strip())
                item_id = jd['item_id']
                item_image_name = jd['item_image_name']
                title = jd.get('title', "")
                item_pvs = jd.get('item_pvs', "")
                cate_name = jd.get('cate_name', "")
                item_pvs = item_pvs.replace("#", "")
                if not item_pvs.endswith(";"):
                    item_pvs += ';'
                item_pvs = " ".join(jieba.cut(item_pvs))
                title = " ".join(jieba.cut(title))
                image_id = f"{item_id}_{self.file_type}"
                image_path = os.path.join(self.image_dir, item_image_name)
                image_h, image_w, num_boxes, boxes, features, cls_prob = 0, 0, 0, 0, 0, 0
                image = cv2.imread(image_path)
                if image is not None:
                    try:
                        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                        image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
                        # 抽取图像特征
                        detection_feature = get_detections_from_image(predictor, image_rgb)
                        if detection_feature is not None:
                            image_h = detection_feature['image_h']
                            image_w = detection_feature['image_w']
                            num_boxes = detection_feature['num_boxes']
                            boxes = detection_feature['boxes']
                            features = detection_feature['features']
                            cls_prob = detection_feature['cls_prob']
                    except cv2.error as e:
                        logger.warning(f"[CV2 ERROR] image_id: {image_id}")
                    except Exception as e:
                        logger.warning(f"[Image ERROR] image_id: {image_id}")
                # 图像与其余模态混合存储
                self.lines.append([item_id, title, item_pvs, cate_name, image_h, image_w, num_boxes, boxes, features, cls_prob])

                ct += 1
                if ct % 5000 == 0:
                    logger.info(f"{file_type}: {ct} images processed")
                    # break

        self.num_lines = len(self.lines)
        if shuffle:
            random.shuffle(self.lines)

        # self.num_file = numfile
        # self.image_feature_file_name = image_feature_file_name
        # self.image_feature_file_name = os.path.join(corpus_path, filetype+'.tsv.%d')
        # print(self.name)
        # if given_file_id:
        #     self.infiles = [self.name % i for i in given_file_id]
        # else:#没给就是所有的
        #     self.infiles = [self.name % i for i in range(self.num_file)]
        # for index, the_file in enumerate(self.infiles):
        #     print(index,':',the_file)#文件排个序
        # self.counts = []
        # self.num_caps = num_caps#
        # if file_type == 'train':
        #     all_df = pd.read_csv('./data/image_lmdb_json/df_train.csv', encoding='utf-8',
        #                          dtype={'image_id': str, 'item_ID': str})#指定类型
        # else:
        #     all_df = pd.read_csv('./data/image_lmdb_json/df_val.csv', encoding='utf-8',
        #                          dtype={'image_id': str, 'item_ID': str})
        #
        # for image_id, pv, caption, category in zip(all_df['image_id'], all_df['pv'], all_df['caption'], all_df['category']):
        #     self.cap_pv_cls[image_id] = (pv, caption, category)

    def __len__(self):
        return self.num_lines

    def __iter__(self):
        for line in self.lines:
            yield line


def serialize(args, dtype):
    predictor = get_predictor(args)
    ds = Conceptual_Caption(args, dtype, predictor)

    if sys.platform.startswith("win"):
        out_file = os.path.join(args.output_dir, f"{dtype}_feat.npz")
        serializer = td.NumpySerializer
    else:
        ds = td.PrefetchDataZMQ(ds, 1)
        out_file = os.path.join(args.output_dir, f"{dtype}_feat.lmdb")
        serializer = td.LMDBSerializer

    if os.path.isfile(out_file):
        os.remove(out_file)

    logger.info(f"{dtype} data length: {len(ds)}")
    try:
        serializer.save(ds, out_file)
    except Exception as e:
        logger.error("[Error] serialization", e)
        # traceback.print_exc()


def main():
    args = get_parser()

    # # step 1: 对应图片重命名
    # for dtype in ["train", "valid"]:
    #     rename_images(args, dtype)
    # logger.info("[Step 1] Finished renaming images")

    # # step 2: 生成lmdb文件
    # for dtype in ["train", "valid"]:
    #     generate_lmdb_df(args, dtype)
    #     generate_lmdb_json(args, dtype)
    # logger.info("[Step 2] Finished generating lmbd files")
    #
    # # step 3: 抽取图像特征
    # for dtype in ["train", "valid"]:
    #     generate_image_features(args, dtype)
    # logger.info("[Step 3] Finished extracting image features")

    # step 4: 抽取图像特征，和其余模态混合，序列化存储
    for dtype in ["train+valid"]:
        serialize(args, dtype)
    logger.info("Finished serializing files")


if __name__ == '__main__':
    main()
