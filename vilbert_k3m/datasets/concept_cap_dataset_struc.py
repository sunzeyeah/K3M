# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import math
import os
import random

import lmdb
import numpy as np
import tensorpack.dataflow as td

import torch
from torch.utils.data import Dataset
from torch.utils.data.sampler import Sampler
import torch.distributed as dist
import sys
import pdb

import msgpack
import msgpack_numpy

msgpack_numpy.patch()

MAX_MSGPACK_LEN = 1000000000

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)s]  %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


def iou(anchors, gt_boxes):
    """
    anchors: (N, 4) ndarray of float
    gt_boxes: (K, 4) ndarray of float
    overlaps: (N, K) ndarray of overlap between boxes and query_boxes
    """
    N = anchors.shape[0]
    K = gt_boxes.shape[0]

    gt_boxes_area = (
            (gt_boxes[:, 2] - gt_boxes[:, 0] + 1) * (gt_boxes[:, 3] - gt_boxes[:, 1] + 1)
    ).reshape(1, K)

    anchors_area = (
            (anchors[:, 2] - anchors[:, 0] + 1) * (anchors[:, 3] - anchors[:, 1] + 1)
    ).reshape(N, 1)

    boxes = np.repeat(anchors.reshape(N, 1, 4), K, axis=1)
    query_boxes = np.repeat(gt_boxes.reshape(1, K, 4), N, axis=0)

    iw = (
            np.minimum(boxes[:, :, 2], query_boxes[:, :, 2])
            - np.maximum(boxes[:, :, 0], query_boxes[:, :, 0])
            + 1
    )
    iw[iw < 0] = 0

    ih = (
            np.minimum(boxes[:, :, 3], query_boxes[:, :, 3])
            - np.maximum(boxes[:, :, 1], query_boxes[:, :, 1])
            + 1
    )
    ih[ih < 0] = 0

    ua = anchors_area + gt_boxes_area - (iw * ih)
    overlaps = iw * ih / ua

    return overlaps


def deserialize_lmdb(ds):
    return msgpack.loads(
        ds[1],
        raw=False,
        max_bin_len=MAX_MSGPACK_LEN,
        max_array_len=MAX_MSGPACK_LEN,
        max_map_len=MAX_MSGPACK_LEN,
        max_str_len=MAX_MSGPACK_LEN,
    )


class InputExample(object):
    """A single training/test example for the language model."""

    def __init__(
            self,
            image_feat=None,
            image_target=None,
            caption=None,
            is_next=None,
            pv=None,  # add
            is_next_pv_v=None,
            is_next_pv_t=None,
            lm_labels=None,
            lm_labels_pv=None,
            image_loc=None,
            num_boxes=None,
            overlaps=None,
    ):
        self.image_feat = image_feat
        self.caption = caption
        self.is_next = is_next  # nextSentence

        self.pv = pv#add
        self.is_next_pv_v=is_next_pv_v
        self.is_next_pv_t = is_next_pv_t

        self.lm_labels = lm_labels  # masked words for language model

        self.lm_labels_pv=lm_labels_pv  # add masked words for language model of pv

        self.image_loc = image_loc
        self.image_target = image_target
        self.num_boxes = num_boxes
        self.overlaps = overlaps


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(
            self,
            input_ids=None,
            input_mask=None,
            segment_ids=None,
            is_next=None,
            lm_label_ids=None,

            input_ids_pv=None,
            input_mask_pv=None,
            segment_ids_pv=None,
            lm_label_ids_pv=None,
            is_next_pv_v=None,
            is_next_pv_t=None,

            image_feat=None,
            image_target=None,
            image_loc=None,
            image_label=None,
            image_mask=None,
            masked_label=None,
            index_p=None,
            index_v=None
    ):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.is_next = is_next
        self.lm_label_ids = lm_label_ids

        self.input_ids_pv = input_ids_pv
        self.input_mask_pv = input_mask_pv
        self.segment_ids_pv = segment_ids_pv
        self.is_next_pv_v = is_next_pv_v
        self.is_next_pv_t = is_next_pv_t
        self.lm_label_ids_pv = lm_label_ids_pv

        self.image_feat = image_feat
        self.image_loc = image_loc
        self.image_label = image_label
        self.image_target = image_target
        self.image_mask = image_mask
        self.masked_label = masked_label
        self.index_p = index_p
        self.index_v = index_v


class ConceptCapLoaderTrain_struc(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the
            GPU, which is faster).
    """
    def __init__(
            self,
            corpus_path,
            file_name,
            tokenizer,
            max_seq_len=32,
            max_seq_len_pv=32,
            max_num_pv=20,
            max_region_len=36,
            v_feature_size=2048,
            v_target_size=1601,
            v_loc_size=5,
            visual_target=0,
            batch_size=512,
            num_workers=25,
            cache=10000,
            rank=-1,
            objective=0,
            visualization=False,
            serializer=td.LMDBSerializer,
    ):
        # if rank != -1:
        #     data_file = os.path.join(corpus_path, file_name.format(rank))
        # else:
        #     data_file = os.path.join(corpus_path, file_name)
        data_file = os.path.join(corpus_path, file_name)
        logger.debug(f"Loading from {data_file}")

        if serializer == td.NumpySerializer:
            buffer = np.load(data_file, allow_pickle=True)['buffer']
            ds = td.DataFromList(buffer, shuffle=True)
        elif serializer == td.LMDBSerializer:
            ds = serializer.load(data_file, shuffle=True)
        else:
            ds = serializer.load(data_file)
        self.num_dataset = len(ds)
        # shuffle data
        # ds = td.LocallyShuffleData(ds, cache)

        preprocess_function = BertPreprocessBatch(tokenizer, max_seq_len=max_seq_len, max_seq_len_pv=max_seq_len_pv, max_num_pv=max_num_pv,
                                                  max_region_len=max_region_len, visual_target=visual_target,
                                                  v_target_size=v_target_size, v_feature_size=v_feature_size,
                                                  v_loc_size=v_loc_size, objective=objective, visualization=visualization)

        ds = td.MapData(ds, preprocess_function)
        if not sys.platform.startswith("win"):
            ds = td.PrefetchDataZMQ(ds, num_workers)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size
        self.num_workers = num_workers

    def __iter__(self):

        for batch in self.ds.get_data():
            item_id, input_ids, input_mask, segment_ids, lm_label_ids, is_next, input_ids_pv, input_mask_pv, segment_ids_pv, \
            lm_label_ids_pv, is_next_pv_v, is_next_pv_t, index_p, index_v, image_feat, image_loc, image_target, image_label, image_mask,\
            masked_label = (
                batch
            )

            batch_size = input_ids.shape[0]
            sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
            sum_count[sum_count == 0] = 1
            g_image_feat = np.sum(image_feat, axis=1) / sum_count
            image_feat = np.concatenate(
                [np.expand_dims(g_image_feat, axis=1), image_feat], axis=1
            )
            image_feat = np.array(image_feat, dtype=np.float32)
            g_image_loc = np.repeat(
                np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0
            )
            image_loc = np.concatenate(
                [np.expand_dims(g_image_loc, axis=1), image_loc], axis=1
            )
            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_next,
                input_ids_pv, input_mask_pv, segment_ids_pv, lm_label_ids_pv, is_next_pv_v, is_next_pv_t,
                image_feat,
                image_loc,
                image_target,
                image_label,
                image_mask,
            )

            yield tuple([torch.tensor(data) for data in batch] + [index_p, index_v, item_id])

    def __len__(self):
        return self.ds.size()


class ConceptCapLoaderVal_struc(object):
    """
    Data loader. Combines a dataset and a sampler, and provides
    single- or multi-process iterators over the dataset.
    Arguments:
        mode (str, required): mode of dataset to operate in, one of ['train', 'val']
        batch_size (int, optional): how many samples per batch to load
            (default: 1).
        shuffle (bool, optional): set to ``True`` to have the data reshuffled
            at every epoch (default: False).
        num_workers (int, optional): how many subprocesses to use for data
            loading. 0 means that the data will be loaded in the main process
            (default: 0)
        cache (int, optional): cache size to use when loading data,
        drop_last (bool, optional): set to ``True`` to drop the last incomplete batch,
            if the dataset size is not divisible by the batch size. If ``False`` and
            the size of dataset is not divisible by the batch size, then the last batch
            will be smaller. (default: False)
        cuda (bool, optional): set to ``True`` and the PyTorch tensors will get preloaded
            to the GPU for you (necessary because this lets us to uint8 conversion on the
            GPU, which is faster).
    """

    def __init__(
            self,
            corpus_path,
            file_name,
            tokenizer,
            max_seq_len=32,
            max_seq_len_pv=32,
            max_num_pv=20,
            max_region_len=36,
            v_feature_size=2048,
            v_target_size=1601,
            v_loc_size=5,
            visual_target=0,
            batch_size=512,
            objective=0,
            visualization=False,
            serializer=td.LMDBSerializer
    ):
        data_file = os.path.join(corpus_path, file_name)
        logger.debug(f"Loading from {data_file}")

        if serializer == td.NumpySerializer:
            buffer = np.load(data_file, allow_pickle=True)['buffer']
            ds = td.DataFromList(buffer, shuffle=False)
        elif serializer == td.LMDBSerializer:
            ds = serializer.load(data_file, shuffle=False)
        else:
            ds = serializer.load(data_file)
        self.num_dataset = len(ds)
        preprocess_function = BertPreprocessBatch(tokenizer, max_seq_len=max_seq_len, max_seq_len_pv=max_seq_len_pv, max_num_pv=max_num_pv,
                                                  max_region_len=max_region_len, visual_target=visual_target,
                                                  v_target_size=v_target_size, v_feature_size=v_feature_size,
                                                  v_loc_size=v_loc_size, visualization=visualization, objective=objective)

        ds = td.MapData(ds, preprocess_function)
        self.ds = td.BatchData(ds, batch_size)
        self.ds.reset_state()

        self.batch_size = batch_size

    def __iter__(self):
        for batch in self.ds.get_data():
            item_id, input_ids, input_mask, segment_ids, lm_label_ids, is_next, input_ids_pv, input_mask_pv, segment_ids_pv, \
            lm_label_ids_pv, is_next_pv_v, is_next_pv_t, index_p, index_v, image_feat, image_loc, image_target, image_label, image_mask, \
            masked_label = (
                batch
            )

            batch_size = input_ids.shape[0]
            sum_count = np.sum(masked_label == 0, axis=1, keepdims=True)
            sum_count[sum_count == 0] = 1
            g_image_feat = np.sum(image_feat, axis=1) / sum_count
            image_feat = np.concatenate(
                [np.expand_dims(g_image_feat, axis=1), image_feat], axis=1
            )
            image_feat = np.array(image_feat, dtype=np.float32)

            g_image_loc = np.repeat(
                np.array([[0, 0, 1, 1, 1]], dtype=np.float32), batch_size, axis=0
            )
            image_loc = np.concatenate(
                [np.expand_dims(g_image_loc, axis=1), image_loc], axis=1
            )

            image_loc = np.array(image_loc, dtype=np.float32)
            g_image_mask = np.repeat(np.array([[1]]), batch_size, axis=0)
            image_mask = np.concatenate([g_image_mask, image_mask], axis=1)

            batch = (
                input_ids,
                input_mask,
                segment_ids,
                lm_label_ids,
                is_next,

                input_ids_pv, input_mask_pv, segment_ids_pv, lm_label_ids_pv, is_next_pv_v, is_next_pv_t,

                image_feat,
                image_loc,
                image_target,
                image_label,
                image_mask,
            )

            yield tuple([torch.tensor(data) for data in batch] + [index_p, index_v, item_id])

    def __len__(self):
        return self.ds.size()


class BertPreprocessBatch(object):
    def __init__(
            self,
            tokenizer,
            max_seq_len=32,
            max_seq_len_pv=32,
            max_num_pv=20,
            max_region_len=36,
            v_feature_size=2048,
            v_target_size=1601,
            v_loc_size=5,
            visual_target=0,
            visualization=False,
            objective=0,
    ):
        self.max_seq_len = max_seq_len
        self.max_seq_len_pv = max_seq_len_pv
        self.max_num_pv = max_num_pv
        self.max_region_len = max_region_len
        self.v_feature_size = v_feature_size
        self.v_target_size = v_target_size
        self.v_loc_size = v_loc_size
        self.tokenizer = tokenizer
        self.visual_target = visual_target
        self.captions = []#[i[1] for i in json.load(open(caption_path, "r"))]
        self.pvs = []#[i[1] for i in json.load(open(pv_path, "r"))]
        self.pvs_len = len(self.pvs)
        self.captions_len = len(self.captions)
        # self.captions = list(json.load(open(caption_path, "r")).values())
        self.visualization = visualization
        self.objective = objective

    def __call__(self, data):
        item_id, caption, pv, category, image_h, image_w, num_boxes, image_location_wp, image_feature_wp, \
        image_target_wp = (
            data
        )

        # Step 1: image processing
        image_feature = np.zeros((self.max_region_len, self.v_feature_size), dtype=np.float32)
        image_target = np.zeros((self.max_region_len, self.v_target_size), dtype=np.float32)
        image_location = np.zeros((self.max_region_len, self.v_loc_size), dtype=np.float32)
        num_boxes = int(num_boxes)
        # calculate the IOU here.
        overlaps = iou(image_location_wp, image_location_wp)
        image_feature[:num_boxes] = image_feature_wp
        image_target[:num_boxes] = image_target_wp
        image_location[:num_boxes, :4] = image_location_wp
        image_location[:, 4] = (
                (image_location[:, 3] - image_location[:, 1])
                * (image_location[:, 2] - image_location[:, 0])
                / (float(image_w) * float(image_h))
        )
        image_location[:, 0] = image_location[:, 0] / float(image_w)
        image_location[:, 1] = image_location[:, 1] / float(image_h)
        image_location[:, 2] = image_location[:, 2] / float(image_w)
        image_location[:, 3] = image_location[:, 3] / float(image_h)
        if self.visual_target == 0:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_target)
        else:
            image_feature = copy.deepcopy(image_feature)
            image_target = copy.deepcopy(image_feature)

        # Step 2: text processing
        label, label_pv_t, label_pv_v = 0, 0, 0
        tokens_caption = self.tokenizer.encode(caption)
        tokens_pv = self.tokenizer.encode(pv)
        logger.debug(f"title:{caption}, tokens caption: {tokens_caption},"
                     f"pv: {pv}, tokens pv: {tokens_pv}")

        # Step 3: transform example to features
        cur_example = InputExample(
            # title
            caption=tokens_caption,
            # item pvs
            pv=tokens_pv,
            is_next=label,
            is_next_pv_v=label_pv_v,
            is_next_pv_t=label_pv_t,
            # image
            image_feat=image_feature,
            image_target=image_target,
            image_loc=image_location,
            num_boxes=num_boxes,
            overlaps=overlaps,
        )
        cur_features = self.convert_example_to_features(cur_example)
        # print(f"current features: {cur_features}")
        return (
            item_id,
            # title tensors
            cur_features.input_ids,
            cur_features.input_mask,
            cur_features.segment_ids,
            cur_features.lm_label_ids,
            cur_features.is_next,
            # pv tensors
            cur_features.input_ids_pv,
            cur_features.input_mask_pv,
            cur_features.segment_ids_pv,
            cur_features.lm_label_ids_pv,
            cur_features.is_next_pv_v,
            cur_features.is_next_pv_t,
            cur_features.index_p,
            cur_features.index_v,
            # image tensors
            cur_features.image_feat,
            cur_features.image_loc,
            cur_features.image_target,
            cur_features.image_label,
            cur_features.image_mask,
            cur_features.masked_label,
        )

    def convert_example_to_features(self, example):
        image_feat = example.image_feat
        tokens = example.caption
        tokens_pv = example.pv
        image_loc = example.image_loc
        image_target = example.image_target
        num_boxes = int(example.num_boxes)
        # is_next = example.is_next
        # is_next_pv_v = example.is_next_pv_v
        # is_next_pv_t = example.is_next_pv_t
        overlaps = example.overlaps

        ## 1. Text Processing
        self._truncate_seq_pair(tokens, self.max_seq_len - 2)
        self._truncate_seq_pair(tokens_pv, self.max_seq_len_pv - 2)
        tokens, tokens_label = self.mask_word(tokens)
        tokens_pv, tokens_label_pv = self.mask_word_pv(tokens_pv)
        # concatenate lm labels and account for CLS, SEP, SEP
        lm_label_ids = [-1] + tokens_label + [-1]
        lm_label_ids_pv = [-1] + tokens_label_pv + [-1]
        tokens = self.tokenizer.add_special_tokens_single_sentence(tokens)
        tokens_pv = self.tokenizer.add_special_tokens_single_sentence(tokens_pv) # [cls_token_id] + token_ids + [sep_token_id].
        index_p, index_v = self.index_pv(tokens_pv) #
        segment_ids = [0] * len(tokens)
        segment_ids_pv = [0] * len(tokens_pv)
        input_ids = tokens  # tokenizer.convert_tokens_to_ids(tokens)
        input_ids_pv = tokens_pv  # tokenizer.convert_tokens_to_ids(tokens)
        # padding
        input_mask = [1] * (len(input_ids))
        input_mask_pv = [1] * (len(input_ids_pv))
        while len(input_ids) < self.max_seq_len:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            lm_label_ids.append(-1)

        while len(input_ids_pv) < self.max_seq_len_pv:
            input_ids_pv.append(0)
            input_mask_pv.append(0)
            segment_ids_pv.append(0)
            lm_label_ids_pv.append(-1)

        assert len(index_p) == len(index_v)

        while len(index_p) < self.max_num_pv:
            index_p.append([0, 0])
        while len(index_v) < self.max_num_pv:
            index_v.append([0, 0])

        # sanity check
        assert len(input_ids) == self.max_seq_len
        assert len(input_mask) == self.max_seq_len
        assert len(segment_ids) == self.max_seq_len
        assert len(lm_label_ids) == self.max_seq_len
        assert len(input_ids_pv) == self.max_seq_len_pv
        assert len(input_mask_pv) == self.max_seq_len_pv
        assert len(segment_ids_pv) == self.max_seq_len_pv
        assert len(lm_label_ids_pv) == self.max_seq_len_pv
        assert len(index_p) == self.max_num_pv
        assert len(index_v) == self.max_num_pv


        ## 2. Image Processing
        masked_label = None
        image_label = []
        image_mask = []
        if num_boxes > 0:
            image_feat, image_loc, image_label, masked_label = self.mask_region(image_feat, image_loc, num_boxes, overlaps)
            # padding
            image_mask = [1] * (num_boxes)
            while len(image_mask) < self.max_region_len:
                image_mask.append(0)
                image_label.append(-1)
            # sanity check
            assert len(image_mask) == self.max_region_len
            assert len(image_label) == self.max_region_len

        return InputFeatures(
            input_ids=np.array(input_ids),
            input_mask=np.array(input_mask),
            segment_ids=np.array(segment_ids),
            lm_label_ids=np.array(lm_label_ids),
            is_next=np.array(example.is_next),
            input_ids_pv=np.array(input_ids_pv),
            input_mask_pv=np.array(input_mask_pv),
            segment_ids_pv=np.array(segment_ids_pv),
            lm_label_ids_pv=np.array(lm_label_ids_pv),
            is_next_pv_v=np.array(example.is_next_pv_v),
            is_next_pv_t=np.array(example.is_next_pv_t),
            image_feat=image_feat,
            image_target=image_target,
            image_loc=image_loc,
            image_label=np.array(image_label),
            image_mask=np.array(image_mask),
            masked_label=masked_label,
            index_p=np.array(index_p),
            index_v=np.array(index_v)
        )

    def _truncate_seq_pair(self, tokens_b, max_length):
        """Truncates a sequence pair in place to the maximum length."""

        # This is a simple heuristic which will always truncate the longer sequence
        # one token at a time. This makes more sense than truncating an equal percent
        # of tokens from each, since if one sequence is very short then each token
        # that's truncated likely contains more information than a longer sequence.
        while True:
            total_length = len(tokens_b)
            if total_length <= max_length:
                break

            tokens_b.pop()

    def mask_word(self, tokens):
        output_label = []

        for i, token in enumerate(tokens):
            prob = random.random()
            # mask token with 15% probability
            if prob < 0.15 and (not self.visualization):
                prob /= 0.15
                # 80% change to mask token
                if prob < 0.8:
                    tokens[i] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
                # 10% change to random token
                elif prob < 0.9:
                    tokens[i] = np.random.randint(len(self.tokenizer))
                # 10% no change
                output_label.append(token)
            # no masking token (will be ignored by loss function later)
            else:
                output_label.append(-1)

        return tokens, output_label
    
    def index_pv(self, tokens):
        # [cls_token_id] + 真token_ids + [sep_token_id].
        #[cls, 6132, 7305, 6199, 131, 2861, 7216, 132, xxx, xxx, 131, xxx, xxx, xxx, 132...
        index_131 = [] #[4,10]
        index_132 = [] #[7,14]
        for i, tok_id in enumerate(tokens):
            if tok_id == 131:
                index_131.append(i)
            if tok_id == 132:
                index_132.append(i)
        if len(index_132) == len(index_131):
            pass
        elif len(index_132) == len(index_131) - 1:
            index_131 = index_131[:-1]
        else:
            index_131 = []
            index_132 = []
        
        index_p = []
        index_v = []
        pv_begin = 1
        for idx131, idx132 in zip(index_131, index_132):
            index_p.append([pv_begin, idx131]) #[[1,4],[8,10]]
            index_v.append([idx131+1, idx132]) #[[5,7],[11,14]]
            pv_begin = idx132 + 1
            if len(index_p) >= self.max_num_pv or len(index_v) >= self.max_num_pv:
                break
        
        return index_p, index_v 

    def mask_word_pv(self, tokens):
        # token_id=131, token=:
        # token_id=132, token=;
        
        index_131 = []
        index_132 = []
        for i, tok_id in enumerate(tokens):#
            if tok_id == 131:
                index_131.append(i)
            if tok_id == 132:
                index_132.append(i)
        
        if len(index_132) == len(index_131)-1:
            index_132.append(len(tokens)) 
        if len(index_132) > 1:
            index_132=index_132[1:]#
            index_131=index_131[1:]#
            
        output_label = [-1]*len(tokens)
        #print(tokens)
        for beg_i, end_i in zip(index_131, index_132):#v index
            for i in range(beg_i+1, end_i):# v
                output_label[i] = tokens[i]
                tokens[i] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        
        return tokens, output_label

        # have_131 = -1
        # have_132 = -1
        # try:
        #     have_131 = tokens.index(131)#3
        #     have_132 = tokens.index(132)#6#need mask 4(have_131+1),5  for i in range(have_131+1,have_132):mask
        # except:
        #     have_132 = len(tokens)
        #
        #
        # output_label = [-1]*len(tokens)
        # if have_131 != -1 and have_132 != -1:
        #     for i in range(have_131+1, have_132):
        #         output_label[i] = tokens[i]
        #         tokens[i] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        #     return tokens, output_label
        #
        # output_label = []
        # for i, token in enumerate(tokens):
        #     #print(tokens)
        #     prob = random.random()
        #     # mask token with 15% probability
        #
        #     # if is_next == 1 and self.objective != 0:
        #     #     prob = 1 # not sample mask
        #     if prob < 0.15 and (not self.visualization):
        #         prob /= 0.15
        #
        #         # 80% randomly change token to mask token
        #         if prob < 0.8:
        #             tokens[i] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)
        #
        #         # 10% randomly change token to random token
        #         elif prob < 0.9:
        #             tokens[i] = np.random.randint(len(self.tokenizer))
        #             # torch.randint(len(tokenizer), labels.shape, dtype=torch.long)
        #
        #         # -> rest 10% randomly keep current token
        #         # append current token to output (we will predict these later)
        #         output_label.append(token)
        #     else:
        #         # no masking token (will be ignored by loss function later)
        #         output_label.append(-1)
        #
        # return tokens, output_label

    def mask_region(self, image_feat, image_loc, num_boxes, overlaps):
        output_label = []
        masked_label = np.zeros((image_feat.shape[0]))  # 全是补齐了的36个
        max_length = len(masked_label)  # 36

        
        if num_boxes < max_length:
            zero_array = np.zeros((num_boxes, max_length - num_boxes))
            overlaps = np.column_stack((overlaps, zero_array))

        for i in range(num_boxes):
            prob = random.random()
            # mask token with 15% probability

            # if is_next == 1 and self.objective != 0:
            #     prob = 1 # if the target is inaligned mask, then not sample mask
            if prob < 0.15 and not self.visualization:
                prob /= 0.15

                # 80% randomly change token to mask token
                if prob < 0.9:
                    image_feat[i] = 0
                # mask the overlap regions into zeros
                # print('concept_cap_dataset overlaps.py\n',overlaps.shape)
                # print('\t\t\tmasked_label',len(masked_label))
                # print('\t\t\tnum_boxes',num_boxes)
                # print('\t\t\timage_feat.shape',image_feat.shape)
                # print('\t\t\tmasked_label1',masked_label)
                # print('\t\t\toverlaps[i]',overlaps[i])
                # print('\t\t\toverlaps.shape',overlaps.shape)
                # masked_label = np.logical_or(masked_label, overlaps[i] > 0.4)#modify
                """上一句长度报错，可以overlaps[i]后补0：和0PADDING的关系肯定也是0吧，
                与其他图关系大于0.4的mask,这样理解，每个box都累计一次"""
                masked_label = np.logical_or(masked_label, overlaps[i] > 0.4)  # modify
                # print('\t\t\tmasked_label2',masked_label)

                # 10% randomly change token to random token
                # elif prob < 0.9:
                # tokens[i] = random.choice(list(tokenizer.vocab.items()))[0]

                # -> rest 10% randomly keep current token
                # append current token to output (we will predict these later)
                output_label.append(1)
            else:
                # no masking token (will be ignored by loss function later)
                output_label.append(-1)

        return image_feat, image_loc, output_label, masked_label
