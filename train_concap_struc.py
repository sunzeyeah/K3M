# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
import logging
import os
import random
from io import open
import sys
import torch
import numpy as np
import tensorpack.dataflow as td
import torch.distributed as dist

# from torch.utils.data import DataLoader, Dataset, RandomSampler
# from torch.utils.data.distributed import DistributedSampler
# from tensorboardX import SummaryWriter
from pytorch_transformers.tokenization_bert import BertTokenizer
from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule
# import vilbert_k3m.utils as utils
from vilbert_k3m.datasets import ConceptCapLoaderTrain_struc, ConceptCapLoaderVal_struc
from vilbert_k3m.vilbert_k3m import BertConfig, BertForMultiModalPreTraining_tri_stru


logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)s]  %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)


# def get_logger(args):
#     logger = logging.getLogger(__name__)
#     # 日志打印到控制台
#     logger.setLevel(logging.INFO)
#
#     # 输出到文件
#     if not os.path.exists('log'):
#         os.mkdir('log')
#     logname = "log/spam-presample_{}.log".format(args.if_pre_sampling)
#     fh = logging.FileHandler(logname)
#     fh.setLevel(logging.INFO)
#
#     # 设置日志格式
#     formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
#     #ch.setFormatter(formatter)
#     fh.setFormatter(formatter)
#     # 将相应的handler添加在logger对象中
#     #logger.addHandler(ch)
#     logger.addHandler(fh)
#
#     return logger


def warmup_linear(x, warmup=0.002):
    """ Specifies a triangular learning rate schedule where peak is reached at `warmup`*`t_total`-th (as provided to BertAdam) training step.
        After `t_total`-th training step, learning rate is zero. """
    if x < warmup:
        return x / warmup
    return max((x - 1.) / (warmup - 1.), 0)


def get_parser():
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--data_dir", required=True, type=str, help="模型训练数据地址")
    parser.add_argument("--output_dir", required=True, type=str, help="The output directory where the model checkpoints will be written.")
    parser.add_argument("--file_name", required=True, type=str, help="模型训练数据文件名")
    parser.add_argument("--model_name", default="bert-base-uncased", type=str, help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-large-uncased, roberta-base",)

    ## Other parameters
    # data, model & config
    parser.add_argument("--pretrained_model_path", default=None, type=str, help="Bert pre-trained model selected in the list: bert-base-uncased, "
             "bert-base-uncased, roberta-base, roberta-large, ")
    parser.add_argument("--config_file", default="bert_base_6layer_6conect.json", type=str, help="The config file which specified the model details.")
    parser.add_argument("--pretrained_model_weights", default="bert-base-uncased_weight_name.json", type=str, help="预训练模型的权重名称文件")
    parser.add_argument("--file_checkpoint", default="", type=str, help="Resume from checkpoint")
    parser.add_argument("--file_state_dict", default="", type=str, help="resume from only model")
    parser.add_argument("--log_steps", default=1, type=int, help="log model training process every n steps")
    # parser.add_argument("--baseline", action="store_true", help="Wheter to use the baseline model (single bert).")
    parser.add_argument("--distributed", action="store_true", help="whether use chunck for parallel training.")
    parser.add_argument("--cache", default=5000,  type=int, help="whether use chunck for parallel training.")
    # training
    parser.add_argument("--do_train", action="store_true", help="是否进行模型训练")
    parser.add_argument("--do_eval", action="store_true", help="是否进行模型验证")
    parser.add_argument("--seed", default=42, type=int, help="random seed for initialization")
    parser.add_argument("--on_memory", action="store_true", help="Whether to load train samples into memory or use disk")
    parser.add_argument("--local_rank", default=-1, type=int, help="local_rank for distributed training on gpus")
    parser.add_argument("--train_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int, help="Total batch size for training.")
    parser.add_argument("--learning_rate", default=1e-4, type=float, help="The initial learning rate for Adam.")
    parser.add_argument("--num_train_epochs", default=6.0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--start_epoch", default=0, type=float, help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda", action="store_true", help="Whether not to use CUDA when available")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of workers in the dataloader.")
    parser.add_argument("--if_pre_sampling", default=1, type=int, help="sampling strategy, 融合策略 0.mean(交互+不交互) 1.sample1(交互,不交互) 2.sample2(交互,不交互)  3.仅交互")
    parser.add_argument("--with_coattention", action="store_true", help="whether pair loss.")
    parser.add_argument("--objective", default=2, type=int, help="which objective to use \
        0: with ICA loss, \
        1: with ICA loss, for the not aligned pair, no masking objective, \
        2: without ICA loss, do not sample negative pair.")
    parser.add_argument("--freeze", default=-1, type=int, help="specify which layer of textual stream of vilbert_k3m need to fixed.")
    # optimization
    parser.add_argument("--warmup_proportion", default=0.1, type=float, help="Proportion of training to perform linear learning rate warmup for. "
             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--gradient_accumulation_steps", default=1, type=int, help="Number of updates steps to accumualte before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument("--loss_img_weight", default=1, type=float, help="weight for image loss")
    parser.add_argument("--fp16", action="store_true", help="Whether to use 16-bit float precision instead of 32-bit")
    parser.add_argument("--apex_fast", action="store_true", help="Whether to use apex to increase training speed")
    parser.add_argument("--loss_scale", default=0, type=float, help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
             "0 (default value): dynamic loss scaling.\n"
             "Positive power of 2: static loss scaling value.\n")
    # NLP model
    parser.add_argument("--do_lower_case", default=True, type=bool, help="Whether to lower case the input text. True for uncased models, False for cased models.")
    parser.add_argument("--max_seq_length", default=36, type=int, help="The maximum total input sequence length after WordPiece tokenization. \n"
             "Sequences longer than this will be truncated, and sequences shorter than this will be padded.")
    parser.add_argument("--max_seq_length_pv", default=128, type=int, help="The maximum total input sequence length after WordPiece tokenization. \n"
             "Sequences longer than this will be truncated, and sequences shorter \n"
             "than this will be padded.")
    parser.add_argument("--max_num_pv", default=20, type=int, help="maximum number of (property, value) pairs")
    # CV Model
    parser.add_argument("--max_region_length", default=36, type=int, help="The maximum region length of a image")
    parser.add_argument("--dynamic_attention", action="store_true", help="whether use dynamic attention for image")
    parser.add_argument("--visual_target", default=0, type=int, help="which target to use for visual branch. \
        0: soft label, \
        1: regress the feature, \
        2: NCE loss.")  # 图片部分具体拟合的对象
    parser.add_argument("--num_negative", default=255, type=int, help="When visutal_target=2, num of negatives to use")

    return parser.parse_args()


def main():
    args = get_parser()

    # logger = get_logger(args)
    # if args.baseline:
    #     from pytorch_pretrained_bert.modeling import BertConfig
    # else:
    #     from vilbert_k3m.vilbert_k3m import BertConfig

    # 设备识别，cpu or cuda
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device(
            "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        )
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend="nccl")

    logger.info(f"device: {device} n_gpu: {n_gpu}, distributed training: {bool(args.local_rank != -1)}, "
                f"16-bits training: {args.fp16}")

    # 设定随机数种子
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    # 判断default gpu
    default_gpu = False
    if dist.is_available() and args.local_rank != -1:
        rank = dist.get_rank()
        if rank == 0:
            default_gpu = True
    else:
        default_gpu = True
    logger.debug(f'default_gpu: {default_gpu}')

    # 生成模型存储路径
    config = BertConfig.from_json_file(os.path.join(args.output_dir, args.config_file))
    model_path = f"k3m_{args.model_name}_{config.num_hidden_layers}l_{config.num_attention_heads}h"
    output_model_path = os.path.join(args.output_dir, model_path)
    if default_gpu:
        if not os.path.exists(args.output_dir):
            os.makedirs(args.output_dir)
        if not os.path.exists(output_model_path):
            os.makedirs(output_model_path)
        # save all the hidden parameters.
        with open(os.path.join(output_model_path, "hyperparamter.txt"), "w") as f:
            print(args, file=f)  # Python 3.x
            print("\n", file=f)
            print(config, file=f)

    if args.visual_target == 0:
        config.v_target_size = 1601
        config.visual_target = args.visual_target
    else:
        config.v_target_size = 2048
        config.visual_target = args.visual_target
    if "roberta" in args.model_name:
        config.model = "roberta"
    if args.freeze > config.t_biattention_id[0]:
        config.fixed_t_layer = config.t_biattention_id[0]
    config.with_coattention = args.with_coattention
    config.dynamic_attention = args.dynamic_attention
    config.if_pre_sampling = args.if_pre_sampling
    config.num_negative = args.num_negative

    train_batch_size = args.train_batch_size // args.gradient_accumulation_steps
    cache = args.cache
    if dist.is_available() and args.local_rank != -1:
        num_replicas = dist.get_world_size()
        train_batch_size = train_batch_size // num_replicas
        args.num_workers = args.num_workers // num_replicas
        cache = cache // num_replicas

    # 读取tokenizer
    tokenizer = BertTokenizer.from_pretrained(args.pretrained_model_path, do_lower_case=args.do_lower_case)
    tokenizer.do_basic_tokenize = False

    # task_names = ["Conceptual_Caption"]
    # task_ids = ["TASK0"]
    # task_num_iters = {"TASK0": train_dataset.num_dataset / args.train_batch_size}
    # logdir = os.path.join(output_model_path, "logs")

    # 创建模型
    if args.pretrained_model_path:
        model = BertForMultiModalPreTraining_tri_stru.from_pretrained(args.pretrained_model_path,
                                                                      config=config, default_gpu=default_gpu)
    else:
        # model = BertForMultiModalPreTraining(config)
        model = BertForMultiModalPreTraining_tri_stru(config)

    if args.fp16 or args.apex_fast:
        assert device == "cuda"
        model.cuda()

    # 冻结部分模型参数
    bert_weight_name = json.load(open(os.path.join(args.output_dir, args.pretrained_model_weights), "r", encoding="utf-8"))
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    if args.freeze != -1:
        bert_weight_name_filtered = []
        for name in bert_weight_name:
            if "embeddings" in name:
                bert_weight_name_filtered.append(name)
            elif "encoder" in name:
                layer_num = name.split(".")[2]
                if int(layer_num) <= args.freeze:
                    bert_weight_name_filtered.append(name)
        # optimizer_grouped_parameters = []
        for key, value in dict(model.named_parameters()).items():
            if key[12:] in bert_weight_name_filtered:
                value.requires_grad = False

        if default_gpu:
            logger.info(f"filtered weight: {bert_weight_name_filtered}")

    # 加载之前训练好的模型（如果有）
    if args.file_state_dict:
        need_model_dict = model.state_dict()
        have_model_state = torch.load(args.file_state_dict, map_location=device)
        new_dict = {}
        for attr in have_model_state:
            if attr.startswith("module."):
                attr = attr.replace("module.", "", 1)#先改名
                if attr in need_model_dict:#需要
                    new_dict[attr] = have_model_state["module."+attr]
            else:
                if attr in need_model_dict:#需要
                    new_dict[attr] = have_model_state[attr]
        need_model_dict.update(new_dict)#更新对应的值
        model.load_state_dict(need_model_dict)

        del have_model_state #这里，手动释放cpu内存...
        del new_dict
        logger.info('Successfully loaded model state dict ...')

    if args.file_checkpoint != "" and os.path.exists(args.file_checkpoint):
        checkpoint = torch.load(args.file_checkpoint, map_location=device)
        new_dict = {}
        logger.debug(f"checkpoint keys: {checkpoint.keys()}")
        for attr in checkpoint["model_state_dict"]:
            if attr.startswith("module."):
                new_dict[attr.replace("module.", "", 1)] = checkpoint["model_state_dict"][attr]
            else:
                new_dict[attr] = checkpoint["model_state_dict"][attr]
        model.load_state_dict(new_dict)
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])  
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        global_step = checkpoint["global_step"]

        del checkpoint
        logger.info('Successfully loaded model checkpoint ...')

    if args.fp16 or args.apex_fast:
        model.half()
    if args.local_rank != -1:
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
            )
        model = DDP(model)
    elif n_gpu > 1:
        model = torch.nn.DataParallel(model)

    # 模型训练
    if args.do_train:
        # 创建data loader
        train_dataset = ConceptCapLoaderTrain_struc(
            args.data_dir,
            args.file_name.format("train+valid"),
            tokenizer,
            max_seq_len=args.max_seq_length,
            max_seq_len_pv=args.max_seq_length_pv,
            max_num_pv=args.max_num_pv,
            max_region_len=args.max_region_length,
            batch_size=train_batch_size,
            visual_target=args.visual_target,
            v_target_size=config.v_target_size,
            num_workers=args.num_workers,
            local_rank=args.local_rank,
            objective=args.objective,
            cache=cache,
            serializer=td.NumpySerializer if sys.platform.startswith("win") else td.LMDBSerializer
        )
        # logger.info(f'Finished preparing train data, total {train_dataset.num_dataset} records')

        if args.do_eval:
            validation_dataset = ConceptCapLoaderVal_struc(
                args.data_dir,
                args.file_name.format("valid"),
                tokenizer,
                max_seq_len=args.max_seq_length,
                max_seq_len_pv=args.max_seq_length_pv,
                max_num_pv=args.max_num_pv,
                max_region_len=args.max_region_length,
                batch_size=args.eval_batch_size,
                visual_target=args.visual_target,
                v_target_size=config.v_target_size,
                objective=args.objective,
                serializer=td.NumpySerializer if sys.platform.startswith("win") else td.LMDBSerializer
            )
            # logger.info(f'Finished preparing valid data, total {validation_dataset.num_dataset} records')

        # optimizer 参数设定
        if not args.pretrained_model_path:
            param_optimizer = list(model.named_parameters())
            optimizer_grouped_parameters = [
                {
                    "params": [
                        p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.01,
                },
                {
                    "params": [
                        p for n, p in param_optimizer if any(nd in n for nd in no_decay)
                    ],
                    "weight_decay": 0.0,
                },
            ]
        else:
            optimizer_grouped_parameters = []
            for key, value in dict(model.named_parameters()).items():
                if value.requires_grad:
                    if key[12:] in bert_weight_name:
                        lr = args.learning_rate * 0.1
                    else:
                        lr = args.learning_rate

                    if any(nd in key for nd in no_decay):
                        optimizer_grouped_parameters += [
                            {"params": [value], "lr": lr, "weight_decay": 0.0}
                        ]

                    if not any(nd in key for nd in no_decay):
                        optimizer_grouped_parameters += [
                            {"params": [value], "lr": lr, "weight_decay": 0.01}
                        ]

            if default_gpu:
                logger.info(f"length of model.named_parameters(): {len(list(model.named_parameters()))}, "
                            f"length of optimizer_grouped_parameters: {len(optimizer_grouped_parameters)}")

        # set different parameters for vision branch and lanugage branch.
        num_train_optimization_steps = int(
            train_dataset.num_dataset
            / args.train_batch_size
            / args.gradient_accumulation_steps
        ) * (args.num_train_epochs - args.start_epoch)
        if args.fp16:
            try:
                # from apex.optimizers import FP16_Optimizer
                from apex.fp16_utils import FP16_Optimizer  # 新版
                # from apex.optimizers import FusedAdam
                import apex.optimizers as apex_optim

            except ImportError:
                raise ImportError(
                    "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training."
                )
            logger.info('Using fp16')
            # optimizer = FusedAdam(optimizer_grouped_parameters,lr=args.learning_rate,bias_correction=False,max_grad_norm=1.0,)#新版没有max_gard_norm
            optimizer = apex_optim.FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, bias_correction=False,
                                             weight_decay=5e-4)
            scheduler = WarmupLinearSchedule(
                optimizer,
                warmup_steps=args.warmup_proportion * num_train_optimization_steps,
                t_total=num_train_optimization_steps,
            )

            if args.loss_scale == 0:
                optimizer = FP16_Optimizer(optimizer, dynamic_loss_scale=True)
            else:
                optimizer = FP16_Optimizer(optimizer, static_loss_scale=args.loss_scale)
        elif args.apex_fast:#add for apex 加速
            from apex import amp
            import apex.optimizers as apex_optim
            logger.info('Using apex_fast')
            optimizer = apex_optim.FusedAdam(optimizer_grouped_parameters, lr=args.learning_rate, bias_correction=False, weight_decay=5e-4)
            scheduler = WarmupLinearSchedule(
                optimizer,
                warmup_steps=args.warmup_proportion * num_train_optimization_steps,
                t_total=num_train_optimization_steps,
            )

            model, optimizer = amp.initialize(model, optimizer, opt_level='O1')  # 这里是字母O
        else:
            logger.info('Using BertAdam')
            optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=args.learning_rate,
                eps=args.adam_epsilon,
                betas=(0.9, 0.98),
            )
            logger.info(f'initial learning rate:{optimizer.param_groups[0]["lr"]}')

            scheduler = WarmupLinearSchedule(
                optimizer,
                warmup_steps=args.warmup_proportion * num_train_optimization_steps,
                t_total=num_train_optimization_steps,
            )

        if not args.no_cuda:
            model.cuda()
        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        if default_gpu:
            logger.info("***** Running training *****")
            logger.info("  Num examples = %d", train_dataset.num_dataset)
            logger.info("  Batch size = %d", args.train_batch_size)
            logger.info("  Num steps = %d", num_train_optimization_steps)

        # 开始训练
        start_iter_id = 0
        global_step = 0
        for epoch in range(int(args.start_epoch), int(args.num_train_epochs)):
            model.train()
            for step, batch in enumerate(train_dataset):
                index_p = torch.tensor(batch[-3])
                index_v = torch.tensor(batch[-2])
                batch = tuple(batch[:-3])
                if not args.no_cuda:
                    index_p = index_p.cuda(device=device, non_blocking=True)
                    index_v = index_v.cuda(device=device, non_blocking=True)
                    batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

                input_ids, input_mask, segment_ids, lm_label_ids, is_next, input_ids_pv, input_mask_pv, \
                segment_ids_pv, lm_label_ids_pv, is_next_pv_v, is_next_pv_t, image_feat, image_loc, \
                image_target, image_label, image_mask = (batch)

                if args.objective == 1:
                    if_replace = is_next + is_next_pv_v + is_next_pv_t
                    image_label = image_label * (if_replace == 0).long().unsqueeze(1)  # 把替换了的对应行变为0
                    image_label[image_label == 0] = -1  # 发生了替换的mask标签还是归于-1
                    # print("image_label",image_label)
                    # print("lm_label_ids",lm_label_ids)
                    # lm_label_ids = lm_label_ids * (is_next == 0).long().unsqueeze(1)
                    lm_label_ids = lm_label_ids * (if_replace == 0).long().unsqueeze(1)
                    lm_label_ids[lm_label_ids == 0] = -1
                    # print("lm_label_ids",lm_label_ids)
                    # print("lm_label_ids_pv",lm_label_ids_pv)
                    lm_label_ids_pv = lm_label_ids_pv * (if_replace == 0).long().unsqueeze(1)
                    lm_label_ids_pv[lm_label_ids_pv == 0] = -1
                    # print("lm_label_ids_pv",lm_label_ids_pv)

                if args.fp16 or args.apex_fast:
                    image_feat = image_feat.half()
                    image_loc = image_loc.half()
                    image_target = image_target.half()

                # import pdb; pdb.set_trace()
                masked_loss_t, masked_loss_v, next_sentence_loss, masked_loss_pv, next_sentence_loss_pv_v, \
                next_sentence_loss_pv_t, next_sentence_loss_t_v_pv, c_initial, c_final, loss_tri = model(
                    input_ids,
                    image_feat,
                    image_loc,
                    segment_ids,
                    input_mask,
                    image_mask,
                    lm_label_ids,
                    image_label,
                    image_target,
                    is_next,
                    output_all_attention_masks=False,  # 默认值False
                    input_ids_pv=input_ids_pv,
                    token_type_ids_pv=segment_ids_pv,  # segnents
                    attention_mask_pv=input_mask_pv,
                    masked_lm_labels_pv=lm_label_ids_pv,
                    next_sentence_label_pv_v=is_next_pv_v,
                    next_sentence_label_pv_t=is_next_pv_t,
                    index_p=index_p,
                    index_v=index_v,
                    device=device
                )
                if args.objective == 2:
                    next_sentence_loss = next_sentence_loss * 0
                    next_sentence_loss_pv_v = next_sentence_loss_pv_v * 0
                    next_sentence_loss_pv_t = next_sentence_loss_pv_t * 0
                    next_sentence_loss_t_v_pv = next_sentence_loss_t_v_pv * 0

                masked_loss_v = masked_loss_v * args.loss_img_weight

                loss = masked_loss_t + masked_loss_v + masked_loss_pv + loss_tri
                #print(masked_loss_pv)
                #print(loss_tri)

                if n_gpu > 1:
                    loss = loss.mean()
                    masked_loss_t = masked_loss_t.mean()
                    masked_loss_v = masked_loss_v.mean()
                    masked_loss_pv = masked_loss_pv.mean()
                    loss_tri = loss_tri.mean()

                if args.local_rank != -1:#分布式训练
                    value_loss = int(loss.cpu().detach().numpy()[0] * 1000) / 1000
                    value_masked_loss_t = int(masked_loss_t.cpu().detach().numpy()[0] * 1000) / 1000
                    value_masked_loss_v = int(masked_loss_v.cpu().detach().numpy()[0] * 1000) / 1000
                    value_masked_loss_pv = int(masked_loss_pv.cpu().detach().numpy()[0] * 1000) / 1000
                    value_loss_tri = int(loss_tri.cpu().detach().numpy()[0] * 1000) / 1000
                else:
                    value_loss = int(loss.cpu().detach().numpy() * 1000) / 1000
                    value_masked_loss_t = int(masked_loss_t.cpu().detach().numpy() * 1000) / 1000
                    value_masked_loss_v = int(masked_loss_v.cpu().detach().numpy() * 1000) / 1000
                    value_masked_loss_pv = int(masked_loss_pv.cpu().detach().numpy() * 1000) / 1000
                    value_loss_tri = int(loss_tri.cpu().detach().numpy() * 1000) / 1000

                if (step + 1) % args.log_steps == 0:
                    logger.info(f"[Epoch-{epoch} Step-{step}] loss: {value_loss} loss_t: {value_masked_loss_t}, "
                                f"loss_v: {value_masked_loss_v}, loss_pv: {value_masked_loss_pv}, loss_tri: {value_loss_tri}")

                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                if args.fp16:
                    optimizer.backward(loss)
                elif args.apex_fast:
                    with amp.scale_loss(loss, optimizer) as scaled_loss:
                        scaled_loss.backward()
                else:
                    loss.backward()

                # 梯度回传
                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    optimizer.zero_grad()
                    global_step += 1

                    # 更新学习率
                    lr_this_step = optimizer.param_groups[0]["lr"]
                    if args.fp16:
                        lr_this_step = args.learning_rate * warmup_linear(
                            global_step / num_train_optimization_steps,
                            args.warmup_proportion,
                            )
                        optimizer.param_groups[0]["lr"] = lr_this_step
                    elif args.apex_fast:
                        scheduler.step()
                    else:#非fp16和apex加速时用
                        scheduler.step()  # 在PyTorch 1.1.0之前的版本，学习率的调整应该被放在optimizer更新之前的，1.1及之后应该位于后面
                    logger.debug(f"lr: {lr_this_step}")

                # # Save a trained model after certain step just model_self
                # if (step+1) % 5000 == 0:#每5000步存储一次
                #     if default_gpu:
                #         # Save a trained model
                #         logger.info("** ** * Saving fine - tuned model ** ** * ")
                #         #print("** ** * Saving fine - tuned model ** ** * ")
                #         model_to_save = (model.module if hasattr(model, "module") else model)  # Only save the model it-self
                #         output_checkpoint = os.path.join(savePath, "K3M_struc-presample_{}.tar".format(args.if_pre_sampling))
                #         output_model_file = os.path.join(savePath, "K3M_struc-presample_{}.bin".format(args.if_pre_sampling))
                #         torch.save(model_to_save.state_dict(), output_model_file)
                #         torch.save(
                #             {
                #                 "model_state_dict": model_to_save.state_dict(),
                #                 "optimizer_state_dict": optimizer.state_dict(),
                #                 "scheduler_state_dict": scheduler.state_dict(),
                #                 "global_step": global_step,
                #             },
                #             output_checkpoint,
                #         )

            # Evaluation per epoch
            if args.do_eval:
                logger.info(f'[Epoch-{epoch}] Starting evaluation ...')
                model.eval()
                torch.set_grad_enabled(False)
                num_batches = len(validation_dataset)

                for step, batch in enumerate(validation_dataset):
                    # image_ids = batch[-1]
                    index_p = torch.tensor(batch[-3])
                    index_v = torch.tensor(batch[-2])
                    batch = tuple(batch[:-3])
                    if not args.no_cuda:
                        index_p = index_p.cuda(device=device, non_blocking=True)
                        index_v = index_v.cuda(device=device, non_blocking=True)
                        batch = tuple(t.cuda(device=device, non_blocking=True) for t in batch)

                    input_ids, input_mask, segment_ids, lm_label_ids, is_next, input_ids_pv, input_mask_pv, segment_ids_pv, \
                    lm_label_ids_pv, is_next_pv_v, is_next_pv_t, image_feat, image_loc, image_target, image_label, image_mask= (batch)

                    # batch_size = input_ids.size(0)
                    if args.fp16 or args.apex_fast:
                        image_feat = image_feat.half()
                        image_loc = image_loc.half()
                        image_target = image_target.half()
                    masked_loss_t, masked_loss_v, next_sentence_loss, masked_loss_pv, next_sentence_loss_pv_v, next_sentence_loss_pv_t, \
                    next_sentence_loss_t_v_pv, c_initial, c_final, loss_tri = model(
                        input_ids,
                        image_feat,
                        image_loc,
                        segment_ids,
                        input_mask,
                        image_mask,
                        lm_label_ids,
                        image_label,
                        image_target,
                        is_next,
                        input_ids_pv=input_ids_pv,
                        token_type_ids_pv=segment_ids_pv,  # segnents
                        attention_mask_pv=input_mask_pv,
                        masked_lm_labels_pv=lm_label_ids_pv,
                        next_sentence_label_pv_v=is_next_pv_v,
                        next_sentence_label_pv_t=is_next_pv_t,
                        index_p=index_p,
                        index_v=index_v,
                        device=device
                    )
                    masked_loss_v = masked_loss_v * args.loss_img_weight
                    loss = masked_loss_t + masked_loss_v + masked_loss_pv + loss_tri

                    if n_gpu > 1:
                        loss = loss.mean()
                        masked_loss_t = masked_loss_t.mean()
                        masked_loss_v = masked_loss_v.mean()
                        masked_loss_pv = masked_loss_pv.mean()
                        loss_tri = loss_tri.mean()

                    if args.local_rank != -1:#分布式训练
                        value_loss = int(loss.cpu().detach().numpy()[0] * 1000) / 1000
                        value_masked_loss_t = int(masked_loss_t.cpu().detach().numpy()[0] * 1000) / 1000
                        value_masked_loss_v = int(masked_loss_v.cpu().detach().numpy()[0] * 1000) / 1000
                        value_masked_loss_pv = int(masked_loss_pv.cpu().detach().numpy()[0] * 1000) / 1000
                        value_loss_tri = int(loss_tri.cpu().detach().numpy()[0] * 1000) / 1000
                    else:
                        value_loss = int(loss.cpu().detach().numpy()* 1000) / 1000
                        value_masked_loss_t = int(masked_loss_t.cpu().detach().numpy() * 1000) / 1000
                        value_masked_loss_v = int(masked_loss_v.cpu().detach().numpy()* 1000) / 1000
                        value_masked_loss_pv = int(masked_loss_pv.cpu().detach().numpy()* 1000) / 1000
                        value_loss_tri = int(loss_tri.cpu().detach().numpy() * 1000) / 1000

                    logger.info(f"[Eval] [Epoch-{epoch}] loss: {value_loss} loss_t: {value_masked_loss_t}, "
                                f"loss_v: {value_masked_loss_v}, loss_pv: {value_masked_loss_pv}, loss_tri: {value_loss_tri}")

                    if default_gpu:
                        sys.stdout.write("%d / %d \r" % (step, num_batches))
                        sys.stdout.flush()

                torch.set_grad_enabled(True)

            # Model saving per epoch
            if default_gpu:
                logger.info(f"[Epoch-{epoch}] saving model")
                model_to_save = (model.module if hasattr(model, "module") else model)  # Only save the model it-self
                output_model_file = os.path.join(output_model_path, f"K3M_struc_presample-{args.if_pre_sampling}_epoch-{epoch}.bin")
                output_checkpoint = os.path.join(output_model_path, f"K3M_struc_presample-{args.if_pre_sampling}_epoch-{epoch}.tar")
                torch.save(model_to_save.state_dict(), output_model_file)
                torch.save(
                    {
                        "model_state_dict": model_to_save.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "global_step": global_step,
                    },
                    output_checkpoint,
                )


if __name__ == "__main__":
    main()
