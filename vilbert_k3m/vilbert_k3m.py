# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import copy
import json
import logging
import math
import os
import random
import tarfile
import tempfile
import sys
from io import open

import torch
from torch import nn
from torch.nn import CrossEntropyLoss, MarginRankingLoss, BCEWithLogitsLoss
import torch.nn.functional as F
# from torch.nn.utils.weight_norm import weight_norm

from .utils import PreTrainedModel
# import pdb
# import numpy as np
# from sklearn import metrics
# import time

logging.basicConfig(
    format="%(asctime)s %(levelname)-4s [%(filename)s:%(lineno)s]  %(message)s",
    datefmt="%Y/%m/%d %H:%M:%S",
    level=logging.INFO
)
logger = logging.getLogger(__name__)

BERT_PRETRAINED_MODEL_ARCHIVE_MAP = {
    "bert-base-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin",
    "bert-large-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-pytorch_model.bin",
    "bert-base-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-pytorch_model.bin",
    "bert-large-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-pytorch_model.bin",
    "bert-base-multilingual-uncased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-uncased-pytorch_model.bin",
    "bert-base-multilingual-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-multilingual-cased-pytorch_model.bin",
    "bert-base-chinese": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese-pytorch_model.bin",
    "bert-base-german-cased": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-german-cased-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-pytorch_model.bin",
    "bert-large-cased-whole-word-masking": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-pytorch_model.bin",
    "bert-large-uncased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-uncased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-large-cased-whole-word-masking-finetuned-squad": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-large-cased-whole-word-masking-finetuned-squad-pytorch_model.bin",
    "bert-base-cased-finetuned-mrpc": "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-cased-finetuned-mrpc-pytorch_model.bin",
    "roberta-base": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-base-pytorch_model.bin",
    "roberta-large": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-pytorch_model.bin",
    "roberta-large-mnli": "https://s3.amazonaws.com/models.huggingface.co/bert/roberta-large-mnli-pytorch_model.bin",
}


def load_tf_weights_in_bert(model, tf_checkpoint_path):
    """ Load tf checkpoints in a pytorch model
    """
    try:
        import re
        import numpy as np
        import tensorflow as tf
    except ImportError:
        print(
            "Loading a TensorFlow models in PyTorch, requires TensorFlow to be installed. Please see "
            "https://www.tensorflow.org/install/ for installation instructions."
        )
        raise
    tf_path = os.path.abspath(tf_checkpoint_path)
    print("Converting TensorFlow checkpoint from {}".format(tf_path))
    # Load weights from TF model
    init_vars = tf.train.list_variables(tf_path)
    names = []
    arrays = []
    for name, shape in init_vars:
        print("Loading TF weight {} with shape {}".format(name, shape))
        array = tf.train.load_variable(tf_path, name)
        names.append(name)
        arrays.append(array)

    for name, array in zip(names, arrays):
        name = name.split("/")
        # adam_v and adam_m are variables used in AdamWeightDecayOptimizer to calculated m and v
        # which are not required for using pretrained model
        if any(n in ["adam_v", "adam_m"] for n in name):
            print("Skipping {}".format("/".join(name)))
            continue
        pointer = model
        for m_name in name:
            if re.fullmatch(r"[A-Za-z]+_\d+", m_name):
                l = re.split(r"_(\d+)", m_name)
            else:
                l = [m_name]
            if l[0] == "kernel" or l[0] == "gamma":
                pointer = getattr(pointer, "weight")
            elif l[0] == "output_bias" or l[0] == "beta":
                pointer = getattr(pointer, "bias")
            elif l[0] == "output_weights":
                pointer = getattr(pointer, "weight")
            else:
                pointer = getattr(pointer, l[0])
            if len(l) >= 2:
                num = int(l[1])
                pointer = pointer[num]
        if m_name[-11:] == "_embeddings":
            pointer = getattr(pointer, "weight")
        elif m_name == "kernel":
            array = np.transpose(array)
        try:
            assert pointer.shape == array.shape
        except AssertionError as e:
            e.args += (pointer.shape, array.shape)
            raise
        print("Initialize PyTorch weight {}".format(name))
        pointer.data = torch.from_numpy(array)
    return model


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


class GeLU(nn.Module):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return gelu(x)


def swish(x):
    return x * torch.sigmoid(x)


ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}


class BertConfig(object):
    """Configuration class to store the configuration of a `BertModel`.
    """

    def __init__(
            self,
            vocab_size_or_config_json_file,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            hidden_act="gelu",
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
            max_position_embeddings=512,
            type_vocab_size=2,
            initializer_range=0.02,
            v_feature_size=2048,
            v_target_size=1601,
            v_hidden_size=768,
            v_num_hidden_layers=3,
            v_num_attention_heads=12,
            v_intermediate_size=3072,
            bi_hidden_size=1024,
            bi_num_attention_heads=16,
            v_attention_probs_dropout_prob=0.1,
            v_hidden_act="gelu",
            v_hidden_dropout_prob=0.1,
            v_initializer_range=0.2,
            v_biattention_id=[0, 1],
            t_biattention_id=[10, 11],
            visual_target=0,
            fast_mode=False,
            fixed_v_layer=0,
            fixed_t_layer=0,
            in_batch_pairs=False,
            fusion_method="mul",
            dynamic_attention=False,
            with_coattention=True,
            objective=0,
            num_negative_image=128,
            num_negative_pv=4,
            margin=1.0,
            model="bert",
            task_specific_tokens=False,
            visualization=False,
            use_image=True,
    ):

        """Constructs BertConfig.

        Args:
            vocab_size_or_config_json_file: Vocabulary size of `inputs_ids` in `BertModel`.
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        assert len(v_biattention_id) == len(t_biattention_id)
        assert max(v_biattention_id) < v_num_hidden_layers
        assert max(t_biattention_id) < num_hidden_layers

        if isinstance(vocab_size_or_config_json_file, str) or (
                sys.version_info[0] == 2
                and isinstance(vocab_size_or_config_json_file, unicode)
        ):
            with open(vocab_size_or_config_json_file, "r", encoding="utf-8") as reader:
                json_config = json.loads(reader.read())
            for key, value in json_config.items():
                self.__dict__[key] = value
        elif isinstance(vocab_size_or_config_json_file, int):
            self.vocab_size = vocab_size_or_config_json_file
            self.hidden_size = hidden_size
            self.num_hidden_layers = num_hidden_layers
            self.num_attention_heads = num_attention_heads
            self.hidden_act = hidden_act
            self.intermediate_size = intermediate_size
            self.hidden_dropout_prob = hidden_dropout_prob
            self.attention_probs_dropout_prob = attention_probs_dropout_prob
            self.max_position_embeddings = max_position_embeddings
            self.type_vocab_size = type_vocab_size
            self.initializer_range = initializer_range
            self.v_feature_size = v_feature_size
            self.v_hidden_size = v_hidden_size
            self.v_num_hidden_layers = v_num_hidden_layers
            self.v_num_attention_heads = v_num_attention_heads
            self.v_intermediate_size = v_intermediate_size
            self.v_attention_probs_dropout_prob = v_attention_probs_dropout_prob
            self.v_hidden_act = v_hidden_act
            self.v_hidden_dropout_prob = v_hidden_dropout_prob
            self.v_initializer_range = v_initializer_range
            self.v_biattention_id = v_biattention_id
            self.t_biattention_id = t_biattention_id
            self.v_target_size = v_target_size
            self.bi_hidden_size = bi_hidden_size
            self.bi_num_attention_heads = bi_num_attention_heads
            self.visual_target = visual_target
            self.fast_mode = fast_mode
            self.fixed_v_layer = fixed_v_layer
            self.fixed_t_layer = fixed_t_layer

            self.model = model
            self.in_batch_pairs = in_batch_pairs
            self.fusion_method = fusion_method
            self.dynamic_attention = dynamic_attention
            self.with_coattention = with_coattention
            self.objective = objective
            self.num_negative_image = num_negative_image
            self.num_negative_pv = num_negative_pv
            self.margin = margin
            self.task_specific_tokens = task_specific_tokens
            self.visualization = visualization
            self.use_image=use_image
        else:
            raise ValueError(
                "First argument must be either a vocabulary size (int)"
                "or the path to a pretrained model config file (str)"
            )

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BertConfig` from a Python dictionary of parameters."""
        config = BertConfig(vocab_size_or_config_json_file=-1)
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BertConfig` from a json file of parameters."""
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


try:
    from apex.normalization.fused_layer_norm import FusedLayerNorm as BertLayerNorm
except ImportError:
    logger.info(
        "Better speed can be achieved with apex installed from https://www.github.com/nvidia/apex ."
    )


    class BertLayerNorm(nn.Module):
        def __init__(self, hidden_size, eps=1e-12):
            """Construct a layernorm module in the TF style (epsilon inside the square root).
            """
            super(BertLayerNorm, self).__init__()
            self.weight = nn.Parameter(torch.ones(hidden_size))
            self.bias = nn.Parameter(torch.zeros(hidden_size))
            self.variance_epsilon = eps

        def forward(self, x):
            u = x.mean(-1, keepdim=True)
            s = (x - u).pow(2).mean(-1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.variance_epsilon)
            return self.weight * x + self.bias


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """

    def __init__(self, config):
        super(BertEmbeddings, self).__init__()

        self.task_specific_tokens = config.task_specific_tokens
        self.word_embeddings = nn.Embedding(
            config.vocab_size, config.hidden_size, padding_idx=0
        )
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        if self.task_specific_tokens:
            self.task_embeddings = nn.Embedding(20, config.hidden_size)

    def forward(self, input_ids, token_type_ids=None, task_ids=None, position_ids=None):

        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = words_embeddings + position_embeddings + token_type_embeddings

        if self.task_specific_tokens:
            task_embeddings = self.task_embeddings(task_ids)
            embeddings = torch.cat(
                [embeddings[:, 0:1], task_embeddings, embeddings[:, 1:]], dim=1
            )

        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class RobertaEmbeddings(BertEmbeddings):
    """
    Same as BertEmbeddings with a tiny tweak for positional embeddings indexing.
    """

    def __init__(self, config):
        super(RobertaEmbeddings, self).__init__(config)
        self.padding_idx = 1

    def forward(self, input_ids, token_type_ids=None, position_ids=None):
        seq_length = input_ids.size(1)
        if position_ids is None:
            # Position numbers begin at padding_idx+1. Padding symbols are ignored.
            # cf. fairseq's `utils.make_positions`
            position_ids = torch.arange(
                self.padding_idx + 1,
                seq_length + self.padding_idx + 1,
                dtype=torch.long,
                device=input_ids.device,
            )
            position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return super(RobertaEmbeddings, self).forward(
            input_ids, token_type_ids=token_type_ids, position_ids=position_ids
        )


class BertSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertSelfAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads)
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.visualization = config.visualization

        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
        else:
            attn_data = None

        return context_layer, attn_data


class BertSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertSelfOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertAttention(nn.Module):
    def __init__(self, config):
        super(BertAttention, self).__init__()
        self.self = BertSelfAttention(config)
        self.output = BertSelfOutput(config)

    def forward(self, input_tensor, attention_mask):
        self_output, attention_probs = self.self(input_tensor, attention_mask)
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertIntermediate(nn.Module):
    def __init__(self, config):
        super(BertIntermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertOutput(nn.Module):
    def __init__(self, config):
        super(BertOutput, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertLayer(nn.Module):
    def __init__(self, config):
        super(BertLayer, self).__init__()
        self.attention = BertAttention(config)
        self.intermediate = BertIntermediate(config)
        self.output = BertOutput(config)

    def forward(self, hidden_states, attention_mask):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertImageSelfAttention(nn.Module):
    def __init__(self, config):
        super(BertImageSelfAttention, self).__init__()
        if config.v_hidden_size % config.v_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.v_hidden_size, config.v_num_attention_heads)
            )
        self.dynamic_attention = config.dynamic_attention
        self.num_attention_heads = config.v_num_attention_heads
        self.attention_head_size = int(
            config.v_hidden_size / config.v_num_attention_heads
        )

        self.visualization = config.visualization

        self.all_head_size = self.num_attention_heads * self.attention_head_size
        self.query = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value = nn.Linear(config.v_hidden_size, self.all_head_size)

        if self.dynamic_attention:
            self.dyLinear_q = nn.Linear(config.hidden_size, self.all_head_size)
            self.dyLinear_k = nn.Linear(config.hidden_size, self.all_head_size)

        self.dropout = nn.Dropout(config.v_attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):

        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        if self.dynamic_attention:
            pool_embedding = (txt_embedding * txt_attention_mask).sum(1)
            pool_embedding = pool_embedding / txt_attention_mask.sum(1)

            # given pool embedding, Linear and Sigmoid layer.
            gate_q = 1 + torch.sigmoid(self.dyLinear_q(pool_embedding))
            gate_k = 1 + torch.sigmoid(self.dyLinear_k(pool_embedding))

            mixed_query_layer = mixed_query_layer * gate_q.unsqueeze(1)
            mixed_key_layer = mixed_key_layer * gate_k.unsqueeze(1)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        if self.visualization:
            attn_data = {
                "attn": attention_probs,
                "queries": query_layer,
                "keys": key_layer,
            }
        else:
            attn_data = None

        return context_layer, attn_data


class BertImageSelfOutput(nn.Module):
    def __init__(self, config):
        super(BertImageSelfOutput, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageAttention(nn.Module):
    def __init__(self, config):
        super(BertImageAttention, self).__init__()
        self.self = BertImageSelfAttention(config)
        self.output = BertImageSelfOutput(config)

    def forward(self, input_tensor, attention_mask, txt_embedding, txt_attention_mask):
        self_output, attention_probs = self.self(
            input_tensor, attention_mask, txt_embedding, txt_attention_mask
        )
        attention_output = self.output(self_output, input_tensor)
        return attention_output, attention_probs


class BertImageIntermediate(nn.Module):
    def __init__(self, config):
        super(BertImageIntermediate, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_intermediate_size)
        if isinstance(config.v_hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.v_hidden_act, unicode)
        ):
            self.intermediate_act_fn = ACT2FN[config.v_hidden_act]
        else:
            self.intermediate_act_fn = config.v_hidden_act

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class BertImageOutput(nn.Module):
    def __init__(self, config):
        super(BertImageOutput, self).__init__()
        self.dense = nn.Linear(config.v_intermediate_size, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.v_hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states


class BertImageLayer(nn.Module):
    def __init__(self, config):
        super(BertImageLayer, self).__init__()
        self.attention = BertImageAttention(config)
        self.intermediate = BertImageIntermediate(config)
        self.output = BertImageOutput(config)

    def forward(self, hidden_states, attention_mask, txt_embedding, txt_attention_mask):
        attention_output, attention_probs = self.attention(
            hidden_states, attention_mask, txt_embedding, txt_attention_mask
        )
        intermediate_output = self.intermediate(attention_output)
        layer_output = self.output(intermediate_output, attention_output)
        return layer_output, attention_probs


class BertBiAttention(nn.Module):
    def __init__(self, config):
        super(BertBiAttention, self).__init__()
        if config.bi_hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.bi_hidden_size, config.bi_num_attention_heads)
            )

        self.visualization = config.visualization
        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(
            config.bi_hidden_size / config.bi_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.v_hidden_size, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)

        self.query2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask=None,
            use_co_attention_mask=False,
    ):

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1
        # if use_co_attention_mask:
        # attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # we can comment this line for single flow.
        attention_scores2 = attention_scores2 + attention_mask2
        # if use_co_attention_mask:
        # attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        attn_data = None

        if self.visualization:
            attn_data = {
                "attn1": attention_probs1,
                "queries1": query_layer2,
                "keys1": key_layer1,
                "attn2": attention_probs2,
                "querues2": query_layer1,
                "keys2": key_layer2,
            }

        return context_layer1, context_layer2, attn_data


class BertBiAttention_two_text(nn.Module):
    def __init__(self, config):
        super(BertBiAttention_two_text, self).__init__()
        if config.hidden_size % config.bi_num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.bi_num_attention_heads)
            )

        self.visualization = config.visualization
        self.num_attention_heads = config.bi_num_attention_heads
        self.attention_head_size = int(
            config.hidden_size / config.bi_num_attention_heads
        )
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        # self.scale = nn.Linear(1, self.num_attention_heads, bias=False)
        # self.scale_act_fn = ACT2FN['relu']

        self.query1 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key1 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value1 = nn.Linear(config.hidden_size, self.all_head_size)
        # self.logit1 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout1 = nn.Dropout(config.v_attention_probs_dropout_prob)

        self.query2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.key2 = nn.Linear(config.hidden_size, self.all_head_size)
        self.value2 = nn.Linear(config.hidden_size, self.all_head_size)
        # self.logit2 = nn.Linear(config.hidden_size, self.num_attention_heads)

        self.dropout2 = nn.Dropout(config.attention_probs_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (
            self.num_attention_heads,
            self.attention_head_size,
        )
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
            self,
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2
    ):

        # for vision input.
        mixed_query_layer1 = self.query1(input_tensor1)
        mixed_key_layer1 = self.key1(input_tensor1)
        mixed_value_layer1 = self.value1(input_tensor1)
        # mixed_logit_layer1 = self.logit1(input_tensor1)

        query_layer1 = self.transpose_for_scores(mixed_query_layer1)
        key_layer1 = self.transpose_for_scores(mixed_key_layer1)
        value_layer1 = self.transpose_for_scores(mixed_value_layer1)
        # logit_layer1 = self.transpose_for_logits(mixed_logit_layer1)

        # for text input:
        mixed_query_layer2 = self.query2(input_tensor2)
        mixed_key_layer2 = self.key2(input_tensor2)
        mixed_value_layer2 = self.value2(input_tensor2)
        # mixed_logit_layer2 = self.logit2(input_tensor2)

        query_layer2 = self.transpose_for_scores(mixed_query_layer2)
        key_layer2 = self.transpose_for_scores(mixed_key_layer2)
        value_layer2 = self.transpose_for_scores(mixed_value_layer2)
        # logit_layer2 = self.transpose_for_logits(mixed_logit_layer2)

        # Take the dot product between "query2" and "key1" to get the raw attention scores for value 1.
        attention_scores1 = torch.matmul(query_layer2, key_layer1.transpose(-1, -2))
        attention_scores1 = attention_scores1 / math.sqrt(self.attention_head_size)
        attention_scores1 = attention_scores1 + attention_mask1
        # if use_co_attention_mask:
        # attention_scores1 = attention_scores1 + co_attention_mask.permute(0,1,3,2)

        # Normalize the attention scores to probabilities.
        attention_probs1 = nn.Softmax(dim=-1)(attention_scores1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs1 = self.dropout1(attention_probs1)

        context_layer1 = torch.matmul(attention_probs1, value_layer1)
        context_layer1 = context_layer1.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape1 = context_layer1.size()[:-2] + (self.all_head_size,)
        context_layer1 = context_layer1.view(*new_context_layer_shape1)

        # Take the dot product between "query1" and "key2" to get the raw attention scores for value 2.
        attention_scores2 = torch.matmul(query_layer1, key_layer2.transpose(-1, -2))
        attention_scores2 = attention_scores2 / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)

        # we can comment this line for single flow.
        attention_scores2 = attention_scores2 + attention_mask2
        # if use_co_attention_mask:
        # attention_scores2 = attention_scores2 + co_attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs2 = nn.Softmax(dim=-1)(attention_scores2)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs2 = self.dropout2(attention_probs2)

        context_layer2 = torch.matmul(attention_probs2, value_layer2)
        context_layer2 = context_layer2.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape2 = context_layer2.size()[:-2] + (self.all_head_size,)
        context_layer2 = context_layer2.view(*new_context_layer_shape2)

        attn_data = None

        if self.visualization:
            attn_data = {
                "attn1": attention_probs1,
                "queries1": query_layer2,
                "keys1": key_layer1,
                "attn2": attention_probs2,
                "querues2": query_layer1,
                "keys2": key_layer2,
            }

        return context_layer1, context_layer2, attn_data


class BertBiOutput(nn.Module):
    def __init__(self, config):
        super(BertBiOutput, self).__init__()

        self.dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.LayerNorm1 = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.q_dense1 = nn.Linear(config.bi_hidden_size, config.v_hidden_size)
        self.q_dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.q_dense2 = nn.Linear(config.bi_hidden_size, config.hidden_size)
        self.q_dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):
        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertBiOutput_two_txt(nn.Module):
    def __init__(self, config):
        super(BertBiOutput_two_txt, self).__init__()

        self.dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm1 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.q_dense1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_dropout1 = nn.Dropout(config.v_hidden_dropout_prob)

        self.dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm2 = BertLayerNorm(config.hidden_size, eps=1e-12)
        self.dropout2 = nn.Dropout(config.hidden_dropout_prob)

        self.q_dense2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.q_dropout2 = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states1, input_tensor1, hidden_states2, input_tensor2):
        context_state1 = self.dense1(hidden_states1)
        context_state1 = self.dropout1(context_state1)

        context_state2 = self.dense2(hidden_states2)
        context_state2 = self.dropout2(context_state2)

        hidden_states1 = self.LayerNorm1(context_state1 + input_tensor1)
        hidden_states2 = self.LayerNorm2(context_state2 + input_tensor2)

        return hidden_states1, hidden_states2


class BertConnectionLayer(nn.Module):
    def __init__(self, config):
        super(BertConnectionLayer, self).__init__()
        self.biattention = BertBiAttention(config)

        self.biOutput = BertBiOutput(config)

        self.v_intermediate = BertImageIntermediate(config)
        self.v_output = BertImageOutput(config)

        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(
            self,
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask=None,
            use_co_attention_mask=False,
    ):
        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2,
            co_attention_mask,
            use_co_attention_mask,
        )

        attention_output1, attention_output2 = self.biOutput(
            bi_output2, input_tensor1, bi_output1, input_tensor2
        )

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs


class BertConnectionLayer_two_text(nn.Module):
    def __init__(self, config):
        super(BertConnectionLayer_two_text, self).__init__()
        self.biattention = BertBiAttention_two_text(config)

        self.biOutput = BertBiOutput_two_txt(config)

        self.v_intermediate = BertIntermediate(config)  # BertImageIntermediate(config)
        self.v_output = BertOutput(config)  # BertImageOutput(config)

        self.t_intermediate = BertIntermediate(config)
        self.t_output = BertOutput(config)

    def forward(
            self,
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2
    ):
        bi_output1, bi_output2, co_attention_probs = self.biattention(
            input_tensor1,
            attention_mask1,
            input_tensor2,
            attention_mask2
        )

        attention_output1, attention_output2 = self.biOutput(
            bi_output2, input_tensor1, bi_output1, input_tensor2
        )

        intermediate_output1 = self.v_intermediate(attention_output1)
        layer_output1 = self.v_output(intermediate_output1, attention_output1)

        intermediate_output2 = self.t_intermediate(attention_output2)
        layer_output2 = self.t_output(intermediate_output2, attention_output2)

        return layer_output1, layer_output2, co_attention_probs


class BertEncoder(nn.Module):
    ''' Need to extract three things:
        (1) text bert layer: BertLayer
        (2) vision bert layer: BertImageLayer
        (3) Bi-Attention: Given the output of two bertlayer, perform bi-directional attention and add on two layers.

    '''
    def __init__(self, config):
        super(BertEncoder, self).__init__()
        self.use_image = config.use_image
        self.FAST_MODE = config.fast_mode
        self.with_coattention = config.with_coattention
        self.v_biattention_id = config.v_biattention_id  # [0, 1, 2, 3, 4, 5],
        self.t_biattention_id = config.t_biattention_id  # [6, 7, 8, 9, 10, 11]
        self.in_batch_pairs = config.in_batch_pairs
        self.fixed_t_layer = config.fixed_t_layer
        self.fixed_v_layer = config.fixed_v_layer

        # bert laysers
        self.layer = nn.ModuleList(
            [BertLayer(config) for _ in range(config.num_hidden_layers)]
        )

        # image bert layers
        if self.use_image:
            self.v_layer = nn.ModuleList(
                [BertImageLayer(config) for _ in range(config.v_num_hidden_layers)]
            )

        if self.with_coattention:
            if self.use_image:
                self.c_layer = nn.ModuleList(
                    [BertConnectionLayer(config) for _ in range(len(config.v_biattention_id))]
                )
                self.c_layer_pv_v = nn.ModuleList(
                    [BertConnectionLayer(config) for _ in range(len(config.v_biattention_id))])

            self.c_layer_pv_t = nn.ModuleList(
                [BertConnectionLayer_two_text(config) for _ in range(len(config.v_biattention_id))])

    def calculate_for_text_img(  
            self,
            txt_embedding,
            image_embedding,
            txt_attention_mask,
            txt_attention_mask2,
            image_attention_mask,
            co_attention_mask=None,
            output_all_encoded_layers=True,
            output_all_attention_masks=False,
    ):

        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []

        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_c = []

        batch_size, num_words, t_hidden_size = txt_embedding.size()
        _, num_regions, v_hidden_size = image_embedding.size()

        use_co_attention_mask = False
        for v_layer_id, t_layer_id in zip(self.v_biattention_id, self.t_biattention_id):

            v_end = v_layer_id
            t_end = t_layer_id

            assert self.fixed_t_layer <= t_end
            assert self.fixed_v_layer <= v_end

            for idx in range(t_start, self.fixed_t_layer):
                with torch.no_grad():
                    txt_embedding, txt_attention_probs = self.layer[idx](
                        txt_embedding, txt_attention_mask
                    )
                    t_start = self.fixed_t_layer
                    if output_all_attention_masks:
                        all_attention_mask_t.append(txt_attention_probs)

            for idx in range(t_start, t_end):
                txt_embedding, txt_attention_probs = self.layer[idx](
                    txt_embedding, txt_attention_mask
                )
                if output_all_attention_masks:
                    all_attention_mask_t.append(txt_attention_probs)

            for idx in range(v_start, self.fixed_v_layer):
                with torch.no_grad():
                    image_embedding, image_attention_probs = self.v_layer[idx](
                        image_embedding,
                        image_attention_mask,
                        txt_embedding,
                        txt_attention_mask2,
                    )
                    v_start = self.fixed_v_layer

                    if output_all_attention_masks:
                        all_attnetion_mask_v.append(image_attention_probs)

            for idx in range(v_start, v_end):
                image_embedding, image_attention_probs = self.v_layer[idx](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask2,
                )

                if output_all_attention_masks:
                    all_attnetion_mask_v.append(image_attention_probs)

            if count == 0 and self.in_batch_pairs:
                # new batch size is the batch_size ^2
                image_embedding = (
                    image_embedding.unsqueeze(0)
                        .expand(batch_size, batch_size, num_regions, v_hidden_size)
                        .contiguous()
                        .view(batch_size * batch_size, num_regions, v_hidden_size)
                )
                image_attention_mask = (
                    image_attention_mask.unsqueeze(0)
                        .expand(batch_size, batch_size, 1, 1, num_regions)
                        .contiguous()
                        .view(batch_size * batch_size, 1, 1, num_regions)
                )

                txt_embedding = (
                    txt_embedding.unsqueeze(1)
                        .expand(batch_size, batch_size, num_words, t_hidden_size)
                        .contiguous()
                        .view(batch_size * batch_size, num_words, t_hidden_size)
                )
                txt_attention_mask = (
                    txt_attention_mask.unsqueeze(1)
                        .expand(batch_size, batch_size, 1, 1, num_words)
                        .contiguous()
                        .view(batch_size * batch_size, 1, 1, num_words)
                )
                co_attention_mask = (
                    co_attention_mask.unsqueeze(1)
                        .expand(batch_size, batch_size, 1, num_regions, num_words)
                        .contiguous()
                        .view(batch_size * batch_size, 1, num_regions, num_words)
                )

            if count == 0 and self.FAST_MODE:
                txt_embedding = txt_embedding.expand(
                    image_embedding.size(0),
                    txt_embedding.size(1),
                    txt_embedding.size(2),
                )
                txt_attention_mask = txt_attention_mask.expand(
                    image_embedding.size(0),
                    txt_attention_mask.size(1),
                    txt_attention_mask.size(2),
                    txt_attention_mask.size(3),
                )

            if self.with_coattention:
                # do the bi attention.
                image_embedding, txt_embedding, co_attention_probs = self.c_layer[
                    count
                ](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask,
                    co_attention_mask,
                    use_co_attention_mask,
                )
                # print('text&img-->count',count)
                # print(co_attention_probs.size())
                # print(co_attention_probs)

                if output_all_attention_masks:
                    all_attention_mask_c.append(co_attention_probs)

            v_start = v_end
            t_start = t_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)

        for idx in range(v_start, len(self.v_layer)):
            image_embedding, image_attention_probs = self.v_layer[idx](
                image_embedding,
                image_attention_mask,
                txt_embedding,
                txt_attention_mask2,
            )

            if output_all_attention_masks:
                all_attnetion_mask_v.append(image_attention_probs)

        for idx in range(t_start, len(self.layer)):
            txt_embedding, txt_attention_probs = self.layer[idx](
                txt_embedding, txt_attention_mask
            )

            if output_all_attention_masks:
                all_attention_mask_t.append(txt_attention_probs)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)

        return (
            all_encoder_layers_t,
            all_encoder_layers_v,
            (all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c),
        )

    def calculate_for_pv_img(  
            self,
            txt_embedding,
            image_embedding,
            txt_attention_mask,
            txt_attention_mask2,
            image_attention_mask,
            co_attention_mask=None,
            output_all_encoded_layers=True,
            output_all_attention_masks=False,
    ):

        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []

        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_c = []

        batch_size, num_words, t_hidden_size = txt_embedding.size()
        _, num_regions, v_hidden_size = image_embedding.size()

        use_co_attention_mask = False
        for v_layer_id, t_layer_id in zip(self.v_biattention_id, self.t_biattention_id):

            v_end = v_layer_id
            t_end = t_layer_id

            assert self.fixed_t_layer <= t_end
            assert self.fixed_v_layer <= v_end

            for idx in range(t_start, self.fixed_t_layer):
                with torch.no_grad():
                    txt_embedding, txt_attention_probs = self.layer[idx](
                        txt_embedding, txt_attention_mask
                    )
                    t_start = self.fixed_t_layer
                    if output_all_attention_masks:
                        all_attention_mask_t.append(txt_attention_probs)

            for idx in range(t_start, t_end):
                txt_embedding, txt_attention_probs = self.layer[idx](
                    txt_embedding, txt_attention_mask
                )
                if output_all_attention_masks:
                    all_attention_mask_t.append(txt_attention_probs)

            for idx in range(v_start, self.fixed_v_layer):
                with torch.no_grad():
                    image_embedding, image_attention_probs = self.v_layer[idx](
                        image_embedding,
                        image_attention_mask,
                        txt_embedding,
                        txt_attention_mask2,
                    )
                    v_start = self.fixed_v_layer

                    if output_all_attention_masks:
                        all_attnetion_mask_v.append(image_attention_probs)

            for idx in range(v_start, v_end):
                image_embedding, image_attention_probs = self.v_layer[idx](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask2,
                )

                if output_all_attention_masks:
                    all_attnetion_mask_v.append(image_attention_probs)

            if count == 0 and self.in_batch_pairs:
                # new batch size is the batch_size ^2
                image_embedding = (
                    image_embedding.unsqueeze(0)
                        .expand(batch_size, batch_size, num_regions, v_hidden_size)
                        .contiguous()
                        .view(batch_size * batch_size, num_regions, v_hidden_size)
                )
                image_attention_mask = (
                    image_attention_mask.unsqueeze(0)
                        .expand(batch_size, batch_size, 1, 1, num_regions)
                        .contiguous()
                        .view(batch_size * batch_size, 1, 1, num_regions)
                )

                txt_embedding = (
                    txt_embedding.unsqueeze(1)
                        .expand(batch_size, batch_size, num_words, t_hidden_size)
                        .contiguous()
                        .view(batch_size * batch_size, num_words, t_hidden_size)
                )
                txt_attention_mask = (
                    txt_attention_mask.unsqueeze(1)
                        .expand(batch_size, batch_size, 1, 1, num_words)
                        .contiguous()
                        .view(batch_size * batch_size, 1, 1, num_words)
                )
                co_attention_mask = (
                    co_attention_mask.unsqueeze(1)
                        .expand(batch_size, batch_size, 1, num_regions, num_words)
                        .contiguous()
                        .view(batch_size * batch_size, 1, num_regions, num_words)
                )

            if count == 0 and self.FAST_MODE:
                txt_embedding = txt_embedding.expand(
                    image_embedding.size(0),
                    txt_embedding.size(1),
                    txt_embedding.size(2),
                )
                txt_attention_mask = txt_attention_mask.expand(
                    image_embedding.size(0),
                    txt_attention_mask.size(1),
                    txt_attention_mask.size(2),
                    txt_attention_mask.size(3),
                )

            if self.with_coattention:
                # do the bi attention.
                image_embedding, txt_embedding, co_attention_probs = self.c_layer_pv_v[
                    count
                ](
                    image_embedding,
                    image_attention_mask,
                    txt_embedding,
                    txt_attention_mask,
                    co_attention_mask,
                    use_co_attention_mask,
                )
                # print('pv&img-->count',count)
                # print(co_attention_probs.size())
                # print(co_attention_probs)

                if output_all_attention_masks:
                    all_attention_mask_c.append(co_attention_probs)

            v_start = v_end
            t_start = t_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(image_embedding)

        for idx in range(v_start, len(self.v_layer)):
            image_embedding, image_attention_probs = self.v_layer[idx](
                image_embedding,
                image_attention_mask,
                txt_embedding,
                txt_attention_mask2,
            )

            if output_all_attention_masks:
                all_attnetion_mask_v.append(image_attention_probs)

        for idx in range(t_start, len(self.layer)):
            txt_embedding, txt_attention_probs = self.layer[idx](
                txt_embedding, txt_attention_mask
            )

            if output_all_attention_masks:
                all_attention_mask_t.append(txt_attention_probs)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(image_embedding)

        return (
            all_encoder_layers_t,
            all_encoder_layers_v,
            (all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c),
        )

    def calculate_for_two_text(  
            self,
            txt_embedding,
            txt_embedding_pv,
            txt_attention_mask,
            txt_attention_mask_pv,
            output_all_encoded_layers=True,
            output_all_attention_masks=False,
    ):

        v_start = 0
        t_start = 0
        count = 0
        all_encoder_layers_t = []
        all_encoder_layers_v = []

        all_attention_mask_t = []
        all_attnetion_mask_v = []
        all_attention_mask_c = []

        batch_size, num_words, t_hidden_size = txt_embedding.size()
        _, num_regions, v_hidden_size = txt_embedding_pv.size()

        use_co_attention_mask = False
        for v_layer_id, t_layer_id in zip(self.t_biattention_id, self.t_biattention_id): 

            v_end = v_layer_id
            t_end = t_layer_id

            assert self.fixed_t_layer <= t_end
            assert self.fixed_t_layer <= v_end 

            for idx in range(t_start, self.fixed_t_layer):
                with torch.no_grad():
                    txt_embedding, txt_attention_probs = self.layer[idx](
                        txt_embedding, txt_attention_mask
                    )
                    t_start = self.fixed_t_layer
                    if output_all_attention_masks:
                        all_attention_mask_t.append(txt_attention_probs)

            for idx in range(t_start, t_end):
                txt_embedding, txt_attention_probs = self.layer[idx](
                    txt_embedding, txt_attention_mask
                )
                if output_all_attention_masks:
                    all_attention_mask_t.append(txt_attention_probs)

            for idx in range(v_start, self.fixed_t_layer):  
                with torch.no_grad():
                    txt_embedding_pv, image_attention_probs = self.layer[idx](
                        txt_embedding_pv,
                        txt_attention_mask_pv,
                        # txt_embedding,
                        # txt_attention_mask2,
                    )
                    v_start = self.fixed_t_layer

                    if output_all_attention_masks:
                        all_attnetion_mask_v.append(image_attention_probs)

            for idx in range(v_start, v_end):
                txt_embedding_pv, image_attention_probs = self.layer[idx](  # v_layer变layer
                    txt_embedding_pv,
                    txt_attention_mask_pv,
                    # txt_embedding,
                    # txt_attention_mask2,
                )

                if output_all_attention_masks:
                    all_attnetion_mask_v.append(image_attention_probs)

            if count == 0 and self.in_batch_pairs:
                # new batch size is the batch_size ^2
                txt_embedding_pv = (
                    txt_embedding_pv.unsqueeze(0)
                        .expand(batch_size, batch_size, num_regions, v_hidden_size)
                        .contiguous()
                        .view(batch_size * batch_size, num_regions, v_hidden_size)
                )
                txt_attention_mask_pv = (
                    txt_attention_mask_pv.unsqueeze(0)
                        .expand(batch_size, batch_size, 1, 1, num_regions)
                        .contiguous()
                        .view(batch_size * batch_size, 1, 1, num_regions)
                )

                txt_embedding = (
                    txt_embedding.unsqueeze(1)
                        .expand(batch_size, batch_size, num_words, t_hidden_size)
                        .contiguous()
                        .view(batch_size * batch_size, num_words, t_hidden_size)
                )
                txt_attention_mask = (
                    txt_attention_mask.unsqueeze(1)
                        .expand(batch_size, batch_size, 1, 1, num_words)
                        .contiguous()
                        .view(batch_size * batch_size, 1, 1, num_words)
                )
                # co_attention_mask = (
                #     co_attention_mask.unsqueeze(1)
                #         .expand(batch_size, batch_size, 1, num_regions, num_words)
                #         .contiguous()
                #         .view(batch_size * batch_size, 1, num_regions, num_words)
                # )

            if count == 0 and self.FAST_MODE:
                txt_embedding = txt_embedding.expand(
                    txt_embedding_pv.size(0),
                    txt_embedding.size(1),
                    txt_embedding.size(2),
                )
                txt_attention_mask = txt_attention_mask.expand(
                    txt_embedding_pv.size(0),
                    txt_attention_mask.size(1),
                    txt_attention_mask.size(2),
                    txt_attention_mask.size(3),
                )

            if self.with_coattention:
                # do the bi attention.
                txt_embedding_pv, txt_embedding, co_attention_probs = self.c_layer_pv_t[
                    count
                ](
                    txt_embedding_pv,
                    txt_attention_mask_pv,
                    txt_embedding,
                    txt_attention_mask,
                    # co_attention_mask,
                    # use_co_attention_mask,
                )
                # print('text&pv-->count',count)
                # print(co_attention_probs.size())
                # print(co_attention_probs)

                if output_all_attention_masks:
                    all_attention_mask_c.append(co_attention_probs)

            v_start = v_end
            t_start = t_end
            count += 1

            if output_all_encoded_layers:
                all_encoder_layers_t.append(txt_embedding)
                all_encoder_layers_v.append(txt_embedding_pv)

        for idx in range(v_start, len(self.layer)):  
            txt_embedding_pv, image_attention_probs = self.layer[idx](
                txt_embedding_pv,
                txt_attention_mask_pv,
                # txt_embedding,
                # txt_attention_mask2,
            )

            if output_all_attention_masks:
                all_attnetion_mask_v.append(image_attention_probs)

        for idx in range(t_start, len(self.layer)):
            txt_embedding, txt_attention_probs = self.layer[idx](
                txt_embedding, txt_attention_mask
            )

            if output_all_attention_masks:
                all_attention_mask_t.append(txt_attention_probs)

        # add the end part to finish.
        if not output_all_encoded_layers:
            all_encoder_layers_t.append(txt_embedding)
            all_encoder_layers_v.append(txt_embedding_pv)

        return (
            all_encoder_layers_t,
            all_encoder_layers_v,
            (all_attention_mask_t, all_attnetion_mask_v, all_attention_mask_c),
        )

    def forward(
            self,
            txt_embedding,  # embedding_output
            image_embedding,  # v_embedding_output
            txt_attention_mask,  # text
            txt_attention_mask2,  # text
            image_attention_mask,  # visaul
            co_attention_mask=None,  # visual

            txt_embedding_pv=None,
            txt_attention_mask_pv=None,  #
            txt_attention_mask2_pv=None,

            output_all_encoded_layers=True,
            output_all_attention_masks=False,
    ):
        if self.use_image:
            (all_encoder_layers_t_with_v, all_encoder_layers_v_with_t, (
            all_attention_mask_t_with_v, all_attnetion_mask_v_with_t,
            all_attention_mask_c_v_t)) = self.calculate_for_text_img(txt_embedding=txt_embedding,
                                                                     image_embedding=image_embedding,
                                                                     txt_attention_mask=txt_attention_mask,
                                                                     txt_attention_mask2=txt_attention_mask2,
                                                                     image_attention_mask=image_attention_mask,
                                                                     co_attention_mask=co_attention_mask,
                                                                     output_all_encoded_layers=output_all_encoded_layers,
                                                                     output_all_attention_masks=output_all_attention_masks, )

            (all_encoder_layers_pv_with_v, all_encoder_layers_v_with_pv, (
            all_attention_mask_pv_with_v, all_attnetion_mask_v_with_pv,
            all_attention_mask_c_v_pv)) = self.calculate_for_pv_img(txt_embedding=txt_embedding_pv,
                                                                    image_embedding=image_embedding,
                                                                    txt_attention_mask=txt_attention_mask_pv,
                                                                    txt_attention_mask2=txt_attention_mask2_pv,
                                                                    image_attention_mask=image_attention_mask,
                                                                    co_attention_mask=co_attention_mask,
                                                                    output_all_encoded_layers=output_all_encoded_layers,
                                                                    output_all_attention_masks=output_all_attention_masks, )
        else:
            all_encoder_layers_t_with_v = None
            all_encoder_layers_v_with_t = None
            all_attention_mask_t_with_v = None
            all_attnetion_mask_v_with_t = None
            all_attention_mask_c_v_t = None
            all_encoder_layers_pv_with_v = None
            all_encoder_layers_v_with_pv = None
            all_attention_mask_pv_with_v = None
            all_attnetion_mask_v_with_pv = None
            all_attention_mask_c_v_pv = None
        
        (all_encoder_layers_t_with_pv, all_encoder_layers_pv_with_t, (
        all_attention_mask_t_with_pv, all_attnetion_mask_pv_with_t,
        all_attention_mask_c_t_pv)) = self.calculate_for_two_text(txt_embedding=txt_embedding,
                                                                  txt_embedding_pv=txt_embedding_pv,
                                                                  txt_attention_mask=txt_attention_mask,
                                                                  txt_attention_mask_pv=txt_attention_mask_pv,
                                                                  output_all_encoded_layers=output_all_encoded_layers,
                                                                  output_all_attention_masks=output_all_attention_masks, )

        return (all_encoder_layers_t_with_v, all_encoder_layers_v_with_t,
                (all_attention_mask_t_with_v, all_attnetion_mask_v_with_t, all_attention_mask_c_v_t)), (
               all_encoder_layers_pv_with_v, all_encoder_layers_v_with_pv,
               (all_attention_mask_pv_with_v, all_attnetion_mask_v_with_pv, all_attention_mask_c_v_pv)), (
               all_encoder_layers_t_with_pv, all_encoder_layers_pv_with_t,
               (all_attention_mask_t_with_pv, all_attnetion_mask_pv_with_t, all_attention_mask_c_t_pv))


class BertTextPooler(nn.Module):
    def __init__(self, config):
        super(BertTextPooler, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertImagePooler(nn.Module):
    def __init__(self, config):
        super(BertImagePooler, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.bi_hidden_size)
        self.activation = nn.ReLU()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor)
        pooled_output = self.activation(pooled_output)
        return pooled_output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        if isinstance(config.hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.hidden_act
        self.LayerNorm = BertLayerNorm(config.hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertImgPredictionHeadTransform(nn.Module):
    def __init__(self, config):
        super(BertImgPredictionHeadTransform, self).__init__()
        self.dense = nn.Linear(config.v_hidden_size, config.v_hidden_size)
        if isinstance(config.hidden_act, str) or (
                sys.version_info[0] == 2 and isinstance(config.hidden_act, unicode)
        ):
            self.transform_act_fn = ACT2FN[config.hidden_act]
        else:
            self.transform_act_fn = config.v_hidden_act
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertLMPredictionHead, self).__init__()
        self.transform = BertPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            bert_model_embedding_weights.size(1),
            bert_model_embedding_weights.size(0),
            bias=False,
        )
        self.decoder.weight = bert_model_embedding_weights
        self.bias = nn.Parameter(torch.zeros(bert_model_embedding_weights.size(0)))

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states) + self.bias
        return hidden_states


class BertOnlyMLMHead(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertOnlyMLMHead, self).__init__()
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)

    def forward(self, sequence_output):
        prediction_scores = self.predictions(sequence_output)
        return prediction_scores


class BertOnlyNSPHead(nn.Module):
    def __init__(self, config):
        super(BertOnlyNSPHead, self).__init__()
        self.seq_relationship = nn.Linear(config.hidden_size, 2)

    def forward(self, pooled_output):
        seq_relationship_score = self.seq_relationship(pooled_output)
        return seq_relationship_score


class BertPreTrainingHeads(nn.Module):
    def __init__(self, config, bert_model_embedding_weights):
        super(BertPreTrainingHeads, self).__init__()
        self.use_image = config.use_image
        self.predictions = BertLMPredictionHead(config, bert_model_embedding_weights)
        self.seq_relationship = nn.Linear(config.hidden_size, 2)
        if self.use_image:
            self.imagePredictions = BertImagePredictionHead(config)
        self.dropout = nn.Dropout(0.1)
        # self.bi_seq_relationship = nn.Linear(config.bi_hidden_size, 2)
        # self.tri_seq_relationship = nn.Linear(config.bi_hidden_size * 3, 2)
        # self.fusion_method = config.fusion_method

    def forward(
            self, sequence_output_t=None, sequence_output_v=None, pooled_output_t=None, pooled_output_v=None,
            sequence_output_pv=None, pooled_output_pv=None,
    ):
        if True:

            if (pooled_output_t is not None) or (pooled_output_v is not None) or (pooled_output_pv is not None):
                if pooled_output_v is None:
                    pooled_output = pooled_output_t + pooled_output_pv
                else:
                    pooled_output = pooled_output_t + pooled_output_pv + pooled_output_v
                pooled_output_t_v_pv = self.dropout(pooled_output)
                seq_relationship_score_pv_v_pv = self.seq_relationship(pooled_output_t_v_pv)
            else:
                seq_relationship_score_pv_v_pv = 0

            if sequence_output_t is not None:
                prediction_scores_t = self.predictions(sequence_output_t)
            else:
                prediction_scores_t = 0

            if sequence_output_pv is not None:
                prediction_scores_pv = self.predictions(sequence_output_pv)
            else:
                prediction_scores_pv = 0

            if sequence_output_v is not None:
                prediction_scores_v = self.imagePredictions(sequence_output_v)
            else:
                prediction_scores_v = 0
            # seq_relationship_score_t_v = self.bi_seq_relationship(pooled_output_t_v)
            # seq_relationship_score_t_pv = self.bi_seq_relationship(pooled_output_t_pv)
            # seq_relationship_score_pv_v = self.bi_seq_relationship(pooled_output_pv_v)

            return prediction_scores_t, prediction_scores_v, prediction_scores_pv, 0, 0, 0, seq_relationship_score_pv_v_pv


class BertImagePredictionHead(nn.Module):
    def __init__(self, config):
        super(BertImagePredictionHead, self).__init__()
        self.transform = BertImgPredictionHeadTransform(config)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(config.v_hidden_size, config.v_target_size)

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        hidden_states = self.decoder(hidden_states)
        return hidden_states


class BertPreTrainedModel(PreTrainedModel):
    """ An abstract class to handle weights initialization and
        a simple interface for dowloading and loading pretrained models.
    """

    config_class = BertConfig
    pretrained_model_archive_map = BERT_PRETRAINED_MODEL_ARCHIVE_MAP
    load_tf_weights = load_tf_weights_in_bert
    base_model_prefix = "bert"

    def __init__(self, *inputs, **kwargs):
        super(BertPreTrainedModel, self).__init__(*inputs, **kwargs)

    def init_weights(self, module):
        """ Initialize the weights.
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, BertLayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()


class BertModel(BertPreTrainedModel):

    base_model_prefix = ""

    def __init__(self, config):
        super(BertModel, self).__init__(config)

        # initilize word embedding
        if config.model == "bert":
            self.embeddings = BertEmbeddings(config)
        elif config.model == "roberta":
            self.embeddings = RobertaEmbeddings(config)

        self.task_specific_tokens = config.task_specific_tokens

        # initlize the vision embedding
        self.v_embeddings = BertImageEmbeddings(config)

        self.encoder = BertEncoder(config)
        self.t_pooler = BertTextPooler(config)
        self.v_pooler = BertImagePooler(config)

        self.apply(self.init_weights)

    def forward(
            self,
            input_txt,  # input_ids
            input_imgs,  # image_feat
            image_loc,  #
            token_type_ids=None,
            attention_mask=None,
            image_attention_mask=None,
            input_txt_pv=None,
            token_type_ids_pv=None,
            attention_mask_pv=None,
            co_attention_mask=None,
            task_ids=None,
            output_all_encoded_layers=False,
            output_all_attention_masks=False,
    ):  
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)

        # add for PV
        if attention_mask_pv is None:
            attention_mask_pv = torch.ones_like(input_txt_pv)
        if token_type_ids_pv is None:
            token_type_ids_pv = torch.zeros_like(input_txt_pv)

        if image_attention_mask is None:
            image_attention_mask = torch.ones(
                input_imgs.size(0), input_imgs.size(1)
            ).type_as(input_txt)

        if self.task_specific_tokens:
            # extend the mask
            mask_tokens = input_txt.new().resize_(input_txt.size(0), 1).fill_(1)
            attention_mask = torch.cat([mask_tokens, attention_mask], dim=1)
            # add for PV
            mask_tokens_pv = input_txt_pv.new().resize_(input_txt_pv.size(0), 1).fill_(1)
            attention_mask_pv = torch.cat([mask_tokens_pv, attention_mask_pv], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask_pv = attention_mask_pv.unsqueeze(1).unsqueeze(2)
        extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)

        extended_attention_mask2 = attention_mask.unsqueeze(2)
        extended_attention_mask2_pv = attention_mask_pv.unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask2 = extended_attention_mask2.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        # add for PV
        extended_attention_mask_pv = extended_attention_mask_pv.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask_pv = (1.0 - extended_attention_mask_pv) * -10000.0
        extended_attention_mask2_pv = extended_attention_mask2_pv.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        extended_image_attention_mask = extended_image_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0

        if co_attention_mask is None:
            co_attention_mask = torch.zeros(
                input_txt.size(0), input_imgs.size(1), input_txt.size(1)
            ).type_as(extended_image_attention_mask)

        extended_co_attention_mask = co_attention_mask.unsqueeze(1)  # 属于图片

        # extended_co_attention_mask = co_attention_mask.unsqueeze(-1)
        extended_co_attention_mask = extended_co_attention_mask * 5.0
        extended_co_attention_mask = extended_co_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        embedding_output = self.embeddings(input_txt, token_type_ids, task_ids)
        embedding_output_pv = self.embeddings(input_txt_pv, token_type_ids_pv, task_ids)
        v_embedding_output = self.v_embeddings(input_imgs, image_loc)
        (encoded_layers_t, encoded_layers_v, all_attention_mask), (
        encoded_layers_pv_with_v, encoded_layers_v_with_pv, all_attention_mask_v_pv), (
        encoded_layers_t_with_pv, encoded_layers_pv_with_t, all_attention_mask_t_pv) = self.encoder(
            embedding_output,  # text
            v_embedding_output,
            extended_attention_mask,  # text
            extended_attention_mask2,  # text
            extended_image_attention_mask,
            extended_co_attention_mask,

            embedding_output_pv,
            extended_attention_mask_pv,
            extended_attention_mask2_pv,

            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_t = encoded_layers_t[-1]
        sequence_output_v = encoded_layers_v[-1]
        pooled_output_t = self.t_pooler(sequence_output_t)
        pooled_output_v = self.v_pooler(sequence_output_v)
        
        sequence_output_pv_with_v = encoded_layers_pv_with_v[-1]
        sequence_output_v_with_pv = encoded_layers_v_with_pv[-1]
        pooled_output_pv_with_v = self.t_pooler(sequence_output_pv_with_v)
        pooled_output_v_with_pv = self.v_pooler(sequence_output_v_with_pv)

        sequence_output_t_with_pv = encoded_layers_t_with_pv[-1]
        sequence_output_pv_with_t = encoded_layers_pv_with_t[-1]
        pooled_output_t_with_pv = self.t_pooler(sequence_output_t_with_pv)
        pooled_output_pv_with_t = self.t_pooler(sequence_output_pv_with_t)

        if not output_all_encoded_layers:
            encoded_layers_t = encoded_layers_t[-1]
            encoded_layers_v = encoded_layers_v[-1]

            encoded_layers_pv_with_v = encoded_layers_pv_with_v[-1]
            encoded_layers_v_with_pv = encoded_layers_v_with_pv[-1]

            encoded_layers_t_with_pv = encoded_layers_t_with_pv[-1]
            encoded_layers_pv_with_t = encoded_layers_pv_with_t[-1]

        return (
            (encoded_layers_t,
             encoded_layers_v,
             pooled_output_t,
             pooled_output_v,
             all_attention_mask),
            (encoded_layers_pv_with_v,
             encoded_layers_v_with_pv,
             pooled_output_pv_with_v,
             pooled_output_v_with_pv,
             all_attention_mask_v_pv),
            (
                encoded_layers_t_with_pv,
                encoded_layers_pv_with_t,
                pooled_output_t_with_pv,
                pooled_output_pv_with_t,
                all_attention_mask_t_pv
            ),
            (
                embedding_output,  
                embedding_output_pv,  
                v_embedding_output, 
            )
        )  


class BertImageEmbeddings(nn.Module):
    """Construct the embeddings from image, spatial location (omit now) and token_type embeddings.
    """

    def __init__(self, config):
        super(BertImageEmbeddings, self).__init__()

        self.image_embeddings = nn.Linear(config.v_feature_size, config.v_hidden_size)
        self.image_location_embeddings = nn.Linear(5, config.v_hidden_size)
        self.LayerNorm = BertLayerNorm(config.v_hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, input_ids, input_loc):
        img_embeddings = self.image_embeddings(input_ids)
        loc_embeddings = self.image_location_embeddings(input_loc)

        # TODO: we want to make the padding_idx == 0, however, with custom initilization, it seems it will have a bias.
        embeddings = self.LayerNorm(img_embeddings + loc_embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings


class ClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.dense = nn.Linear(config.hidden_size * 2, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        self.out_proj = nn.Linear(config.hidden_size, 2)

    def forward(self, features, **kwargs):
        # x = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(features)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)

        return x


class BertForMultiModalPreTraining_tri_stru(BertPreTrainedModel):
    """BERT model with multi modal pre-training heads.
    """

    def __init__(self, config):
        super(BertForMultiModalPreTraining_tri_stru, self).__init__(config)
        # whether to use image mode
        self.use_image = config.use_image
        # Bert Model Tri
        if config.model == "bert":
            self.embeddings = BertEmbeddings(config)
        elif config.model == "roberta":
            self.embeddings = RobertaEmbeddings(config)
        self.task_specific_tokens = config.task_specific_tokens
        self.encoder = BertEncoder(config)
        self.t_pooler = BertTextPooler(config)
        if self.use_image:
            self.v_embeddings = BertImageEmbeddings(config)
            self.v_pooler = BertImagePooler(config)

        # Bert Head Tri
        self.cls = BertPreTrainingHeads(config, self.embeddings.word_embeddings.weight)

        self.if_pre_sampling = config.if_pre_sampling
        # if self.if_pre_sampling == 0:
        #     logger.info('fusion strategy 0, individual + interactive -- mean')
        # elif self.if_pre_sampling == 1:
        #     logger.info('fusion strategy 1, individual + interactive -- hard')
        # elif self.if_pre_sampling == 2:
        #     logger.info('fusion strategy 2, individual + interactive -- soft')
        # elif self.if_pre_sampling == 3:
        #     logger.info('fusion strategy 3, interactive')
        # else:
        #     pass

        if self.use_image:
            num_modes = 3
            self.map_individual_to_bi = nn.Linear(config.hidden_size, config.bi_hidden_size)
            self.map_bi_to_individual = nn.Linear(config.bi_hidden_size, config.hidden_size)
            # image scores
            self.score_self_v = nn.Linear(config.bi_hidden_size * num_modes, config.bi_hidden_size)#.to(devices[0])
            self.score_cross1_v = nn.Linear(config.bi_hidden_size * num_modes, config.bi_hidden_size)#.to(devices[0])
            self.score_cross2_v = nn.Linear(config.bi_hidden_size * num_modes, config.bi_hidden_size)#.to(devices[0])
            self.soft_v = nn.Linear(config.bi_hidden_size * num_modes, config.bi_hidden_size)#.to(devices[0])
            self.visual_target = config.visual_target
            self.num_negative_image = config.num_negative_image
            if self.visual_target == 0:
                self.vis_criterion = nn.KLDivLoss(reduction="none")
            elif self.visual_target == 1:
                self.vis_criterion = nn.MSELoss(reduction="none")
            elif self.visual_target == 2:
                self.vis_criterion = CrossEntropyLoss()
        else:
            num_modes = 2

        # title scores
        self.score_self_t = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[1])
        self.score_cross1_t = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[1])
        self.score_cross2_t = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[1])
        self.soft_t = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[1])
        # pv scores
        self.score_self_pv = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[2])
        self.score_cross1_pv = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[2])
        self.score_cross2_pv = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[2])
        self.soft_pv = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[2])

        self.apply(self.init_weights)

        self.num_negative_pv = config.num_negative_pv
        self.loss_mlm = CrossEntropyLoss(ignore_index=-1)
        # structure aggregation module
        self.struc_w1 = nn.Linear(config.hidden_size * 3, config.hidden_size)#.to(devices[3])
        self.struc_w2 = nn.Linear(config.hidden_size, 1)#.to(devices[3])
        self.struc_w3 = nn.Linear(config.hidden_size, config.hidden_size)#.to(devices[3])
        # self.struc_w_loss = nn.Linear(config.hidden_size, 2)#.to(devices[3])
        # self.loss_fct_struc = CrossEntropyLoss(ignore_index=-1)
        self.loss_lpm = MarginRankingLoss(margin=config.margin)

        self.tie_weights()

    def tie_weights(self):
        """ Make sure we are sharing the input and output embeddings.
            Export to TorchScript can't handle parameter sharing so we are cloning them instead.
        """
        self._tie_or_clone_weights(
            self.cls.predictions.decoder, self.embeddings.word_embeddings
        )

    def pre_sampling(self, individual_pooled=None, pooled_output_c1=None, pooled_output_c2=None,
                     modality=None):  # modality = v,t,pv

        individual_pooled = F.relu(individual_pooled)
        pooled_output_c1 = F.relu(pooled_output_c1)
        pooled_output_c2 = F.relu(pooled_output_c2)
        feature_list = (individual_pooled, pooled_output_c1, pooled_output_c2)
        if modality == 'v':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_v(torch.cat(feature_list, 1))), dim=1)
            alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_v(torch.cat(feature_list, 1))), dim=1)
            alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_v(torch.cat(feature_list, 1))), dim=1)
        elif modality == 't':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_t(torch.cat(feature_list, 1))), dim=1)
            alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_t(torch.cat(feature_list, 1))), dim=1)
            alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_t(torch.cat(feature_list, 1))), dim=1)
        elif modality == 'pv':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_pv(torch.cat(feature_list, 1))), dim=1)
            alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_pv(torch.cat(feature_list, 1))), dim=1)
            alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_pv(torch.cat(feature_list, 1))), dim=1)

        ak = torch.cat((alpha_s, alpha_c1, alpha_c2), 1) 
        a_index = F.gumbel_softmax(ak, hard=True, dim=1) 
        pooled_output = individual_pooled * (a_index[:, 0, :].squeeze(dim=1)) + pooled_output_c1 * (
            a_index[:, 1, :].squeeze(dim=1)) + pooled_output_c2 * (a_index[:, 2, :].squeeze(dim=1))
        return pooled_output

    def pre_sampling_sequence_soft(self, individual_sequence=None, sequence_c1=None, sequence_c2=None,
                                   modality=None):  # modality = v,t,pv
        individual_sequence = F.relu(individual_sequence)
        sequence_c1 = F.relu(sequence_c1)
        sequence_c2 = F.relu(sequence_c2)  # 16,36,1024
        feature_list = (individual_sequence, sequence_c1, sequence_c2)
        if modality == 'v':
            alpha_s = F.sigmoid(self.score_self_v(torch.cat(feature_list, 2)))
            alpha_c1 = F.sigmoid(self.score_cross1_v(torch.cat(feature_list, 2)))
            alpha_c2 = F.sigmoid(self.score_cross2_v(torch.cat(feature_list, 2)))
            # print(alpha_s.size())
            # print(individual_sequence.size())
            sequence_output = self.soft_v(
                torch.cat((individual_sequence * alpha_s, sequence_c1 * alpha_c1, sequence_c2 * alpha_c2), 2))

        elif modality == 't':
            alpha_s = F.sigmoid(self.score_self_t(torch.cat(feature_list, 2)))
            alpha_c1 = F.sigmoid(self.score_cross1_t(torch.cat(feature_list, 2)))
            alpha_c2 = F.sigmoid(self.score_cross2_t(torch.cat(feature_list, 2)))
            sequence_output = self.soft_t(
                torch.cat((individual_sequence * alpha_s, sequence_c1 * alpha_c1, sequence_c2 * alpha_c2), 2))

        elif modality == 'pv':
            alpha_s = F.sigmoid(self.score_self_pv(torch.cat(feature_list, 2)))
            alpha_c1 = F.sigmoid(self.score_cross1_pv(torch.cat(feature_list, 2)))
            alpha_c2 = F.sigmoid(self.score_cross2_pv(torch.cat(feature_list, 2)))
            sequence_output = self.soft_pv(
                torch.cat((individual_sequence * alpha_s, sequence_c1 * alpha_c1, sequence_c2 * alpha_c2), 2))

        return sequence_output

    def pre_sampling_sequence(self, individual_sequence, sequence_c1=None, sequence_c2=None,
                              modality=None):  # modality = v,t,pv
        if individual_sequence is None:
            return None

        individual_sequence = F.relu(individual_sequence)
        if sequence_c1 is not None:
            sequence_c1 = F.relu(sequence_c1)
        if sequence_c2 is not None:
            sequence_c2 = F.relu(sequence_c2)  # 16,36,1024
        feature_list = tuple(seq for seq in [individual_sequence, sequence_c1, sequence_c2] if seq is not None)
        alpha_s, alpha_c1, alpha_c2 = None, None, None
        if modality == 'v':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_v(torch.cat(feature_list, 2))), dim=2)
            if sequence_c1 is not None:
                alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_v(torch.cat(feature_list, 2))), dim=2)
            if sequence_c2 is not None:
                alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_v(torch.cat(feature_list, 2))), dim=2)
        elif modality == 't':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_t(torch.cat(feature_list, 2))), dim=2)
            if sequence_c1 is not None:
                alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_t(torch.cat(feature_list, 2))), dim=2)
            if sequence_c2 is not None:
                alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_t(torch.cat(feature_list, 2))), dim=2)
        elif modality == 'pv':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_pv(torch.cat(feature_list, 2))), dim=2)
            if sequence_c1 is not None:
                alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_pv(torch.cat(feature_list, 2))), dim=2)
            if sequence_c2 is not None:
                alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_pv(torch.cat(feature_list, 2))), dim=2)

        alphas = tuple(alpha for alpha in [alpha_s, alpha_c1, alpha_c2] if alpha is not None)
        ak = torch.cat(alphas, 2)  #
        a_index = F.gumbel_softmax(ak, hard=True, dim=2)  #
        i = 0
        sequence_output = individual_sequence * (a_index[:, :, i, :].squeeze(dim=2))
        if alpha_c1 is not None:
            i += 1
            sequence_output += sequence_c1 * (a_index[:, :, i, :].squeeze(dim=2))
        if alpha_c2 is not None:
            i += 1
            sequence_output += sequence_c2 * (a_index[:, :, i, :].squeeze(dim=2))

        return sequence_output

    def get_sequence_pooled_output_final(self,
                                         sequence_output_t_with_v, sequence_output_v_with_t,
                                         sequence_output_pv_with_v, sequence_output_v_with_pv,
                                         sequence_output_t_with_pv, sequence_output_pv_with_t,
                                         individual_txt, individual_pv, individual_v):
        if self.if_pre_sampling == 1:  # hard
            sequence_output_v = self.pre_sampling_sequence(individual_v, sequence_output_v_with_t, sequence_output_v_with_pv,
                                                           modality='v')  # 1024
            sequence_output_t = self.pre_sampling_sequence(individual_txt, sequence_output_t_with_v, sequence_output_t_with_pv,
                                                           modality='t')  # 768
            sequence_output_pv = self.pre_sampling_sequence(individual_pv, sequence_output_pv_with_v,
                                                            sequence_output_pv_with_t, modality='pv')  # 768
        elif self.if_pre_sampling == 0:  # mean
            sequence_output_v = (individual_v + sequence_output_v_with_t + sequence_output_v_with_pv) / 3
            sequence_output_t = (individual_txt + sequence_output_t_with_v + sequence_output_t_with_pv) / 3
            sequence_output_pv = (individual_pv + sequence_output_pv_with_v + sequence_output_pv_with_t) / 3
        elif self.if_pre_sampling == 2:  # soft
            sequence_output_v = self.pre_sampling_sequence_soft(individual_v, sequence_output_v_with_t,
                                                                sequence_output_v_with_pv, modality='v')  # 1024
            sequence_output_t = self.pre_sampling_sequence_soft(individual_txt, sequence_output_t_with_v,
                                                                sequence_output_t_with_pv, modality='t')  # 768
            sequence_output_pv = self.pre_sampling_sequence_soft(individual_pv, sequence_output_pv_with_v,
                                                                 sequence_output_pv_with_t, modality='pv')  # 768
        else:  # no fusoin
            sequence_output_v = (sequence_output_v_with_t + sequence_output_v_with_pv) / 2
            sequence_output_t = (sequence_output_t_with_v + sequence_output_t_with_pv) / 2
            sequence_output_pv = (sequence_output_pv_with_v + sequence_output_pv_with_t) / 2

        if self.use_image:
            pooled_output_v = self.map_bi_to_individual(torch.mean(sequence_output_v[:, 1:, :], dim=1))  # 1024-768
        else:
            pooled_output_v = None
        pooled_output_t = torch.mean(sequence_output_t[:, 1:, :], dim=1)  # 768
        pooled_output_pv = torch.mean(sequence_output_pv[:, 1:, :], dim=1)  # 768

        return sequence_output_v, sequence_output_t, sequence_output_pv, pooled_output_v, pooled_output_t, pooled_output_pv

    def structure_aggregator(self, c_initial, sequence_output_pv, index_p, index_v, device):
        ''' Compute 3 values:
            (1) initial entity embedding = pooled image embedding + pooled title embedding + pooled knowledge graph embedding
            (2) final entity embedding = initial entity embedding + attention-weighted triplets embeddings, aka Structure Aggregation Module
            (3) Link Prediction Modeling (LPM) loss

        :param pooled_output_v: pooled image embedding
        :param pooled_output_t: pooled text embedding
        :param pooled_output_pv: pooled knowledge graph embedding
        :param sequence_output_pv: sequence knowledge graph embeddings (for LPM loss)
        :param index_p: index of knowledge graph sequence
        :param index_v: index of image

        :return:
            c_initial: initial entity embedding
            c_final: final entity embedding
            loss_struct: LPM loss
        '''
        # the number of attention heads of structure aggregator in paper is 8, here is 1.
        # We found using 1 attetion head has similar performance as using 8 heads, and it's more efficient.
        
        #--------  structure aggregate module ------------
        property_vecs = []
        value_vecs = []
        for i in range(sequence_output_pv.shape[0]):# item
            property_vecs.append([])
            value_vecs.append([])
            for j in range(index_p.shape[1]):# p
                if index_p[i, j, 0] == 0:
                    break
                p = torch.mean(sequence_output_pv[i, :, :].index_select(dim=0, index=index_p[i, j, ]), dim=0)##[768] , keepdim=True
                v = torch.mean(sequence_output_pv[i, :, :].index_select(dim=0, index=index_v[i, j, :]), dim=0)
                property_vecs[i].append(p)
                value_vecs[i].append(v)
                if j == 0:
                    t = torch.unsqueeze(self.struc_w1(torch.cat((c_initial[i], p, v), dim=0)), 0)
                else:
                    t = torch.cat((t, torch.unsqueeze(self.struc_w1(torch.cat((c_initial[i], p, v), dim=0)), 0)), dim=0)

            try:
                b = self.struc_w2(F.leaky_relu(t))
            except:
                t = torch.unsqueeze(c_initial[i], 0)
                b = self.struc_w2(F.leaky_relu(t))

            # attention
            atten = F.softmax(b, dim=0)
            
            if i == 0:
                c_final = torch.unsqueeze(c_initial[i] + self.struc_w3(torch.sum(atten*t, dim=0)), 0)
                # c_final_neg = torch.unsqueeze(c_initial[i+1] + self.struc_w3(torch.sum(atten*t, dim=0)), 0)#错位构造负样本
            else:
                c_final = torch.cat((c_final, torch.unsqueeze(c_initial[i] + self.struc_w3(torch.sum(atten*t, dim=0)), 0)), dim=0)
                # c_final_neg = torch.cat((
                #     c_final_neg, torch.unsqueeze(c_initial[(i+1) % sequence_output_pv.shape[0]] + self.struc_w3(torch.sum(atten*t, dim=0)), 0)), dim=0)

        # 计算LPM loss
        positive_norms = torch.tensor([], device=device)
        negative_norms = torch.tensor([], device=device)
        for i in range(c_final.shape[0]):
            final_entity_vec = c_final[i]
            for j, (property_vec, value_vec) in enumerate(zip(property_vecs[i], value_vecs[i])):
                # 负样本类型一：随机替换实体, <ec',  property, value>
                num_negative_entities = self.num_negative_pv // 2
                candidates = [k for k in range(c_final.shape[0]) if k != i]
                if len(candidates) > 0:
                    positive_norm = torch.norm(final_entity_vec + property_vec - value_vec)
                    index_negative_entities = random.sample(candidates, min(len(candidates), num_negative_entities))
                    for k in index_negative_entities:
                        negative_entity_vec = c_final[k]
                        negative_norm = torch.norm(negative_entity_vec + property_vec - value_vec)
                        positive_norms = torch.cat((positive_norms, positive_norm.unsqueeze(0)))
                        negative_norms = torch.cat((negative_norms, negative_norm.unsqueeze(0)))

                # 负样本类型二：随机替换值, <ec,  property, value'>
                num_negative_values = self.num_negative_pv - num_negative_entities
                candidates = [k for k in range(len(property_vecs[i])) if k != j]
                if len(candidates) > 0:
                    positive_norm = torch.norm(final_entity_vec + property_vec - value_vec)
                    index_negative_values = random.sample(candidates, min(len(candidates), num_negative_values))
                    for k in index_negative_values:
                        negative_value_vec = value_vecs[i][k]
                        negative_norm = torch.norm(final_entity_vec + property_vec - negative_value_vec)
                        positive_norms = torch.cat((positive_norms, positive_norm.unsqueeze(0)))
                        negative_norms = torch.cat((negative_norms, negative_norm.unsqueeze(0)))

        # struc_label = torch.tensor([1]*sequence_output_pv.shape[0]+[0]*sequence_output_pv.shape[0], device=device)
        # logits = self.struc_w_loss(torch.cat((c_final, c_final_neg), dim=0))
        struc_label = torch.ones(positive_norms.shape[-1], device=device)
        loss_struc = self.loss_lpm(positive_norms, negative_norms, struc_label)
        logger.debug(f"LPM loss: {loss_struc}")

        return c_final, loss_struc

    def bert_tri(self,
                 input_txt,  # input_ids
                 input_imgs,  # image_feat
                 image_loc,  #
                 token_type_ids=None,
                 attention_mask=None,
                 image_attention_mask=None,
                 input_txt_pv=None,
                 token_type_ids_pv=None,
                 attention_mask_pv=None,
                 co_attention_mask=None,
                 task_ids=None,
                 output_all_encoded_layers=False,
                 output_all_attention_masks=False,):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if attention_mask_pv is None:
            attention_mask_pv = torch.ones_like(input_txt_pv)
        if token_type_ids_pv is None:
            token_type_ids_pv = torch.zeros_like(input_txt_pv)
        if image_attention_mask is None and self.use_image:
            image_attention_mask = torch.ones(
                input_imgs.size(0), input_imgs.size(1)
            ).type_as(input_txt)

        if self.task_specific_tokens:
            # extend the mask
            mask_tokens = input_txt.new().resize_(input_txt.size(0), 1).fill_(1)
            attention_mask = torch.cat([mask_tokens, attention_mask], dim=1)
            # add for PV
            mask_tokens_pv = input_txt_pv.new().resize_(input_txt_pv.size(0), 1).fill_(1)
            attention_mask_pv = torch.cat([mask_tokens_pv, attention_mask_pv], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask_pv = attention_mask_pv.unsqueeze(1).unsqueeze(2)


        extended_attention_mask2 = attention_mask.unsqueeze(2)
        extended_attention_mask2_pv = attention_mask_pv.unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask2 = extended_attention_mask2.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        # add for PV
        extended_attention_mask_pv = extended_attention_mask_pv.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask_pv = (1.0 - extended_attention_mask_pv) * -10000.0
        extended_attention_mask2_pv = extended_attention_mask2_pv.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        if self.use_image:
            extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_image_attention_mask = extended_image_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0
            if co_attention_mask is None:
                co_attention_mask = torch.zeros(
                    input_txt.size(0), input_imgs.size(1), input_txt.size(1)
                ).type_as(extended_image_attention_mask)
            extended_co_attention_mask = co_attention_mask.unsqueeze(1)  # 属于图片
            # extended_co_attention_mask = co_attention_mask.unsqueeze(-1)
            extended_co_attention_mask = extended_co_attention_mask * 5.0
            extended_co_attention_mask = extended_co_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            v_embedding_output = self.v_embeddings(input_imgs, image_loc)
        else:
            extended_image_attention_mask = None
            extended_co_attention_mask = None
            v_embedding_output = None

        embedding_output = self.embeddings(input_txt, token_type_ids, task_ids)
        embedding_output_pv = self.embeddings(input_txt_pv, token_type_ids_pv, task_ids)

        (encoded_layers_t_with_v, encoded_layers_v_with_t, all_attention_mask), (
            encoded_layers_pv_with_v, encoded_layers_v_with_pv, all_attention_mask_v_pv), (
            encoded_layers_t_with_pv, encoded_layers_pv_with_t, all_attention_mask_t_pv) = self.encoder(
            embedding_output,  # text
            v_embedding_output,
            extended_attention_mask,  # text
            extended_attention_mask2,  # text
            extended_image_attention_mask,
            extended_co_attention_mask,

            embedding_output_pv,
            extended_attention_mask_pv,
            extended_attention_mask2_pv,

            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_t_with_pv = encoded_layers_t_with_pv[-1]
        sequence_output_pv_with_t = encoded_layers_pv_with_t[-1]
        pooled_output_t_with_pv = self.t_pooler(sequence_output_t_with_pv)
        pooled_output_pv_with_t = self.t_pooler(sequence_output_pv_with_t)

        if self.use_image:
            sequence_output_t_with_v = encoded_layers_t_with_v[-1]
            pooled_output_t_with_v = self.t_pooler(sequence_output_t_with_v)
            sequence_output_v_with_t = encoded_layers_v_with_t[-1]
            pooled_output_v_with_t = self.v_pooler(sequence_output_v_with_t)
            sequence_output_pv_with_v = encoded_layers_pv_with_v[-1]
            sequence_output_v_with_pv = encoded_layers_v_with_pv[-1]
            pooled_output_pv_with_v = self.t_pooler(sequence_output_pv_with_v)
            pooled_output_v_with_pv = self.v_pooler(sequence_output_v_with_pv)
        else:
            pooled_output_t_with_v = None
            pooled_output_v_with_t = None
            pooled_output_pv_with_v = None
            pooled_output_v_with_pv = None

        if not output_all_encoded_layers:
            encoded_layers_t_with_pv = encoded_layers_t_with_pv[-1]
            encoded_layers_pv_with_t = encoded_layers_pv_with_t[-1]
            if self.use_image:
                encoded_layers_t_with_v = encoded_layers_t_with_v[-1]
                encoded_layers_v_with_t = encoded_layers_v_with_t[-1]
                encoded_layers_pv_with_v = encoded_layers_pv_with_v[-1]
                encoded_layers_v_with_pv = encoded_layers_v_with_pv[-1]
            else:
                encoded_layers_t_with_v = None
                encoded_layers_v_with_t = None
                encoded_layers_pv_with_v = None
                encoded_layers_v_with_pv = None

        return (
            (encoded_layers_t_with_v,
             encoded_layers_v_with_t,
             pooled_output_t_with_v,
             pooled_output_v_with_t,
             all_attention_mask),
            (encoded_layers_pv_with_v,
             encoded_layers_v_with_pv,
             pooled_output_pv_with_v,
             pooled_output_v_with_pv,
             all_attention_mask_v_pv),
            (encoded_layers_t_with_pv,
             encoded_layers_pv_with_t,
             pooled_output_t_with_pv,
             pooled_output_pv_with_t,
             all_attention_mask_t_pv),
            (embedding_output,
             embedding_output_pv,
             v_embedding_output)
        )

    def forward(
            self,
            input_ids,  
            image_feat,
            image_loc,
            token_type_ids=None,  # segnents
            attention_mask=None,  # input_mask
            image_attention_mask=None,  # image_mask
            masked_lm_labels=None,  # lm_label_ids
            image_label=None,  # image_label
            image_target=None,  # image_target
            next_sentence_label=None,  # is_next, title
            output_all_attention_masks=False, 
            # pv
            input_ids_pv=None,
            token_type_ids_pv=None,  # segnents
            attention_mask_pv=None,
            masked_lm_labels_pv=None,
            next_sentence_label_pv_v=None,
            next_sentence_label_pv_t=None,
            index_p=None,
            index_v=None,
            device=None
    ):
        (sequence_output_t_with_v, sequence_output_v_with_t, pooled_output_t_with_v, pooled_output_v_with_t, all_attention_mask_t_v), (
        sequence_output_pv_with_v, sequence_output_v_with_pv, pooled_output_pv_with_v, pooled_output_v_with_pv,
        all_attention_mask_v_pv), (
        sequence_output_t_with_pv, sequence_output_pv_with_t, pooled_output_t_with_pv, pooled_output_pv_with_t,
        all_attention_mask_t_pv), (individual_txt, individual_pv, individual_v) = self.bert_tri(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            input_ids_pv,
            token_type_ids_pv,
            attention_mask_pv,
            output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_v, sequence_output_t, sequence_output_pv, pooled_output_v, pooled_output_t, pooled_output_pv = self.get_sequence_pooled_output_final(
            sequence_output_t_with_v, sequence_output_v_with_t,
            sequence_output_pv_with_v, sequence_output_v_with_pv,
            sequence_output_t_with_pv, sequence_output_pv_with_t,
            individual_txt, individual_pv, individual_v)

        # initial item embedding
        if pooled_output_v is not None:
            c_initial = (pooled_output_v + pooled_output_t + pooled_output_pv) / 3 #[batch_size,768]
        else:
            c_initial = (pooled_output_t + pooled_output_pv) / 2
        c_final, loss_struc = self.structure_aggregator(c_initial, sequence_output_pv, index_p, index_v, device)
        
        if True:  
            prediction_scores_t, prediction_scores_v, prediction_scores_pv, seq_relationship_score, seq_relationship_score_t_pv, \
            seq_relationship_score_pv_v, seq_relationship_score_t_v_pv = self.cls(
                sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v, sequence_output_pv,
                pooled_output_pv,
            )
        else:
            prediction_scores_t, prediction_scores_v, seq_relationship_score = self.cls(
                sequence_output_t, sequence_output_v, pooled_output_t, pooled_output_v
            )

        if (
                masked_lm_labels is not None
                and next_sentence_label is not None
                and image_target is not None
        ):
            if self.use_image:
                prediction_scores_v = prediction_scores_v[:, 1:]
                if self.visual_target == 1:
                    img_loss = self.vis_criterion(prediction_scores_v, image_target)
                    masked_img_loss = torch.sum(
                        img_loss * (image_label == 1).unsqueeze(2).float()
                    ) / max(
                        torch.sum((image_label == 1).unsqueeze(2).expand_as(img_loss)), 1
                    )
                elif self.visual_target == 0:
                    img_loss = self.vis_criterion(
                        F.log_softmax(prediction_scores_v, dim=2), image_target
                    )

                    masked_img_loss = torch.sum(
                        img_loss * (image_label == 1).unsqueeze(2).float()
                    ) / max(torch.sum((image_label == 1)), 0)
                elif self.visual_target == 2:
                    # generate negative sampled index.
                    num_negative = self.num_negative_image
                    num_across_batch = int(self.num_negative * 0.7)
                    num_inside_batch = int(self.num_negative * 0.3)

                    batch_size, num_regions, _ = prediction_scores_v.size()
                    assert batch_size != 0
                    # random negative across batches.
                    row_across_index = input_ids.new(
                        batch_size, num_regions, num_across_batch
                    ).random_(0, batch_size - 1)
                    col_across_index = input_ids.new(
                        batch_size, num_regions, num_across_batch
                    ).random_(0, num_regions)

                    for i in range(batch_size - 1):
                        row_across_index[i][row_across_index[i] == i] = batch_size - 1
                    final_across_index = row_across_index * num_regions + col_across_index

                    # random negative inside batches.
                    row_inside_index = input_ids.new(
                        batch_size, num_regions, num_inside_batch
                    ).zero_()
                    col_inside_index = input_ids.new(
                        batch_size, num_regions, num_inside_batch
                    ).random_(0, num_regions - 1)

                    for i in range(batch_size):
                        row_inside_index[i] = i
                    for i in range(num_regions - 1):
                        col_inside_index[:, i, :][col_inside_index[:, i, :] == i] = (
                                num_regions - 1
                        )
                    final_inside_index = row_inside_index * num_regions + col_inside_index

                    final_index = torch.cat((final_across_index, final_inside_index), dim=2)

                    # Let's first sample where we need to compute.
                    predict_v = prediction_scores_v[image_label == 1]
                    neg_index_v = final_index[image_label == 1]

                    flat_image_target = image_target.view(batch_size * num_regions, -1)
                    # we also need to append the target feature at the begining.
                    negative_v = flat_image_target[neg_index_v]
                    positive_v = image_target[image_label == 1]
                    sample_v = torch.cat((positive_v.unsqueeze(1), negative_v), dim=1)

                    # calculate the loss.
                    score = torch.bmm(sample_v, predict_v.unsqueeze(2)).squeeze(2)
                    masked_img_loss = self.vis_criterion(
                        score, input_ids.new(score.size(0)).zero_()
                    )
            else:
                masked_img_loss = torch.zeros(1, device=device)

            masked_lm_loss = self.loss_mlm(
                prediction_scores_t.view(-1, self.config.vocab_size),
                masked_lm_labels.view(-1).to(device=device, dtype=torch.long, non_blocking=True)
            )

            if True:  # pv对的损失，和title的很像
                masked_lm_loss_pv = self.loss_mlm(
                    prediction_scores_pv.view(-1, self.config.vocab_size),
                    masked_lm_labels_pv.view(-1).to(device=device, dtype=torch.long, non_blocking=True)
                )

                next_sentence_label_pv_t_v = 1 - 1 * (
                        (next_sentence_label + next_sentence_label_pv_v + next_sentence_label_pv_t) == 0)
                next_sentence_loss_t_v_pv = self.loss_mlm(seq_relationship_score_t_v_pv.view(-1, 2),
                                                          next_sentence_label_pv_t_v.view(-1).to(device=device, dtype=torch.long, non_blocking=True)
                )

                return (
                    masked_lm_loss.unsqueeze(0),
                    masked_img_loss.unsqueeze(0),
                    # next_sentence_loss.unsqueeze(0),
                    0,
                    masked_lm_loss_pv.unsqueeze(0),
                    0,  # next_sentence_loss_pv_v.unsqueeze(0),
                    0,  # next_sentence_loss_pv_t.unsqueeze(0),
                    next_sentence_loss_t_v_pv.unsqueeze(0),
                    c_initial,
                    c_final,
                    loss_struc.unsqueeze(0),
                )

            return (
                masked_lm_loss.unsqueeze(0),
                masked_img_loss.unsqueeze(0),
                next_sentence_loss.unsqueeze(0),
            )
        else:
            return (
                prediction_scores_t,
                prediction_scores_v,
                seq_relationship_score,
                all_attention_mask,
            )


class K3MForItemAlignment(BertPreTrainedModel):
    """K3M model finetune for item alignment
    """

    def __init__(self, config):
        super(K3MForItemAlignment, self).__init__(config)
        self.use_image = config.use_image
        self.loss_type = config.loss_type
        # Bert Model Tri
        if config.model == "bert":
            self.embeddings = BertEmbeddings(config)
        elif config.model == "roberta":
            self.embeddings = RobertaEmbeddings(config)
        self.task_specific_tokens = config.task_specific_tokens
        if self.use_image:
            self.v_embeddings = BertImageEmbeddings(config)
            self.v_pooler = BertImagePooler(config)
        self.encoder = BertEncoder(config)
        self.t_pooler = BertTextPooler(config)

        self.if_pre_sampling = config.if_pre_sampling
        # if self.if_pre_sampling == 0:
        #     logger.info('fusion strategy 0, individual + interactive -- mean')
        # elif self.if_pre_sampling == 1:
        #     logger.info('fusion strategy 1, individual + interactive -- hard')
        # elif self.if_pre_sampling == 2:
        #     logger.info('fusion strategy 2, individual + interactive -- soft')
        # elif self.if_pre_sampling == 3:
        #     logger.info('fusion strategy 3, interactive')
        # else:
        #     pass

        if self.use_image:
            num_modes = 3
            self.map_individual_to_bi = nn.Linear(config.hidden_size, config.bi_hidden_size)
            self.map_bi_to_individual = nn.Linear(config.bi_hidden_size, config.hidden_size)
            # image scores
            self.score_self_v = nn.Linear(config.bi_hidden_size * num_modes, config.bi_hidden_size)#.to(devices[0])
            self.score_cross1_v = nn.Linear(config.bi_hidden_size * num_modes, config.bi_hidden_size)#.to(devices[0])
            self.score_cross2_v = nn.Linear(config.bi_hidden_size * num_modes, config.bi_hidden_size)#.to(devices[0])
            self.soft_v = nn.Linear(config.bi_hidden_size * num_modes, config.bi_hidden_size)#.to(devices[0])
            self.visual_target = config.visual_target
            self.num_negative_image = config.num_negative_image
            if self.visual_target == 0:
                self.vis_criterion = nn.KLDivLoss(reduction="none")
            elif self.visual_target == 1:
                self.vis_criterion = nn.MSELoss(reduction="none")
            elif self.visual_target == 2:
                self.vis_criterion = CrossEntropyLoss()
        else:
            num_modes = 2

        # title scores
        self.score_self_t = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[1])
        self.score_cross1_t = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[1])
        self.score_cross2_t = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[1])
        self.soft_t = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[1])
        # pv scores
        self.score_self_pv = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[2])
        self.score_cross1_pv = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[2])
        self.score_cross2_pv = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[2])
        self.soft_pv = nn.Linear(config.hidden_size * num_modes, config.hidden_size)#.to(devices[2])
        # self.soft_pv = nn.Linear(config.hidden_size * 2, 2)#.to(devices[2])
        # classification head
        if self.loss_type == "ce":
            self.classifier = ClassificationHead(config)

        self.apply(self.init_weights)
        # self.visual_target = config.visual_target
        # self.num_negative_image = config.num_negative_image
        # self.num_negative_pv = config.num_negative_pv
        if self.loss_type == "ce":
            self.loss_fct = CrossEntropyLoss()
            self.softmax = torch.nn.Softmax()
        elif self.loss_type == "cosine":
            self.loss_fct = nn.CosineEmbeddingLoss(margin=0.0)
        else:
            logger.error("Unsupported type of loss function")
        # structure aggregation module
        self.struc_w1 = nn.Linear(config.hidden_size * 3, config.hidden_size)#.to(devices[3])
        self.struc_w2 = nn.Linear(config.hidden_size, 1)#.to(devices[3])
        self.struc_w3 = nn.Linear(config.hidden_size, config.hidden_size)#.to(devices[3])
        # self.struc_w_loss = nn.Linear(config.hidden_size, 2)#.to(devices[3])
        # self.loss_fct_struc = CrossEntropyLoss(ignore_index=-1)
        # self.loss_lpm = MarginRankingLoss(margin=config.margin)
        self.cosine = nn.CosineSimilarity()

        # if self.visual_target == 0:
        #     self.vis_criterion = nn.KLDivLoss(reduction="none")
        # elif self.visual_target == 1:
        #     self.vis_criterion = nn.MSELoss(reduction="none")
        # elif self.visual_target == 2:
        #     self.vis_criterion = CrossEntropyLoss()

        # self.tie_weights()

    # def tie_weights(self):
    #     """ Make sure we are sharing the input and output embeddings.
    #         Export to TorchScript can't handle parameter sharing so we are cloning them instead.
    #     """
    #     self._tie_or_clone_weights(
    #         self.cls.predictions.decoder, self.embeddings.word_embeddings
    #     )

    def pre_sampling(self, individual_pooled=None, pooled_output_c1=None, pooled_output_c2=None,
                     modality=None):  # modality = v,t,pv

        individual_pooled = F.relu(individual_pooled)
        pooled_output_c1 = F.relu(pooled_output_c1)
        pooled_output_c2 = F.relu(pooled_output_c2)
        feature_list = (individual_pooled, pooled_output_c1, pooled_output_c2)
        if modality == 'v':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_v(torch.cat(feature_list, 1))), dim=1)
            alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_v(torch.cat(feature_list, 1))), dim=1)
            alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_v(torch.cat(feature_list, 1))), dim=1)
        elif modality == 't':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_t(torch.cat(feature_list, 1))), dim=1)
            alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_t(torch.cat(feature_list, 1))), dim=1)
            alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_t(torch.cat(feature_list, 1))), dim=1)
        elif modality == 'pv':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_pv(torch.cat(feature_list, 1))), dim=1)
            alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_pv(torch.cat(feature_list, 1))), dim=1)
            alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_pv(torch.cat(feature_list, 1))), dim=1)

        ak = torch.cat((alpha_s, alpha_c1, alpha_c2), 1)
        a_index = F.gumbel_softmax(ak, hard=True, dim=1)
        pooled_output = individual_pooled * (a_index[:, 0, :].squeeze(dim=1)) + pooled_output_c1 * (
            a_index[:, 1, :].squeeze(dim=1)) + pooled_output_c2 * (a_index[:, 2, :].squeeze(dim=1))
        return pooled_output

    def pre_sampling_sequence_soft(self, individual_sequence=None, sequence_c1=None, sequence_c2=None,
                                   modality=None):  # modality = v,t,pv
        individual_sequence = F.relu(individual_sequence)
        sequence_c1 = F.relu(sequence_c1)
        sequence_c2 = F.relu(sequence_c2)  # 16,36,1024
        feature_list = (individual_sequence, sequence_c1, sequence_c2)
        if modality == 'v':
            alpha_s = F.sigmoid(self.score_self_v(torch.cat(feature_list, 2)))
            alpha_c1 = F.sigmoid(self.score_cross1_v(torch.cat(feature_list, 2)))
            alpha_c2 = F.sigmoid(self.score_cross2_v(torch.cat(feature_list, 2)))
            # print(alpha_s.size())
            # print(individual_sequence.size())
            sequence_output = self.soft_v(
                torch.cat((individual_sequence * alpha_s, sequence_c1 * alpha_c1, sequence_c2 * alpha_c2), 2))

        elif modality == 't':
            alpha_s = F.sigmoid(self.score_self_t(torch.cat(feature_list, 2)))
            alpha_c1 = F.sigmoid(self.score_cross1_t(torch.cat(feature_list, 2)))
            alpha_c2 = F.sigmoid(self.score_cross2_t(torch.cat(feature_list, 2)))
            sequence_output = self.soft_t(
                torch.cat((individual_sequence * alpha_s, sequence_c1 * alpha_c1, sequence_c2 * alpha_c2), 2))

        elif modality == 'pv':
            alpha_s = F.sigmoid(self.score_self_pv(torch.cat(feature_list, 2)))
            alpha_c1 = F.sigmoid(self.score_cross1_pv(torch.cat(feature_list, 2)))
            alpha_c2 = F.sigmoid(self.score_cross2_pv(torch.cat(feature_list, 2)))
            sequence_output = self.soft_pv(
                torch.cat((individual_sequence * alpha_s, sequence_c1 * alpha_c1, sequence_c2 * alpha_c2), 2))

        return sequence_output

    def pre_sampling_sequence(self, individual_sequence, sequence_c1=None, sequence_c2=None,
                              modality=None):  # modality = v,t,pv
        if individual_sequence is None:
            return None

        individual_sequence = F.relu(individual_sequence)
        if sequence_c1 is not None:
            sequence_c1 = F.relu(sequence_c1)
        if sequence_c2 is not None:
            sequence_c2 = F.relu(sequence_c2)  # 16,36,1024
        feature_list = tuple(seq for seq in [individual_sequence, sequence_c1, sequence_c2] if seq is not None)
        alpha_s, alpha_c1, alpha_c2 = None, None, None
        if modality == 'v':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_v(torch.cat(feature_list, 2))), dim=2)
            if sequence_c1 is not None:
                alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_v(torch.cat(feature_list, 2))), dim=2)
            if sequence_c2 is not None:
                alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_v(torch.cat(feature_list, 2))), dim=2)
        elif modality == 't':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_t(torch.cat(feature_list, 2))), dim=2)
            if sequence_c1 is not None:
                alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_t(torch.cat(feature_list, 2))), dim=2)
            if sequence_c2 is not None:
                alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_t(torch.cat(feature_list, 2))), dim=2)
        elif modality == 'pv':
            alpha_s = torch.unsqueeze(F.sigmoid(self.score_self_pv(torch.cat(feature_list, 2))), dim=2)
            if sequence_c1 is not None:
                alpha_c1 = torch.unsqueeze(F.sigmoid(self.score_cross1_pv(torch.cat(feature_list, 2))), dim=2)
            if sequence_c2 is not None:
                alpha_c2 = torch.unsqueeze(F.sigmoid(self.score_cross2_pv(torch.cat(feature_list, 2))), dim=2)

        alphas = tuple(alpha for alpha in [alpha_s, alpha_c1, alpha_c2] if alpha is not None)
        ak = torch.cat(alphas, 2)  #
        a_index = F.gumbel_softmax(ak, hard=True, dim=2)  #
        i = 0
        sequence_output = individual_sequence * (a_index[:, :, i, :].squeeze(dim=2))
        if alpha_c1 is not None:
            i += 1
            sequence_output += sequence_c1 * (a_index[:, :, i, :].squeeze(dim=2))
        if alpha_c2 is not None:
            i += 1
            sequence_output += sequence_c2 * (a_index[:, :, i, :].squeeze(dim=2))

        return sequence_output

    def get_sequence_pooled_output_final(self,
                                         sequence_output_t_with_v, sequence_output_v_with_t,
                                         sequence_output_pv_with_v, sequence_output_v_with_pv,
                                         sequence_output_t_with_pv, sequence_output_pv_with_t,
                                         individual_txt, individual_pv, individual_v):
        if self.if_pre_sampling == 1:  # hard
            sequence_output_v = self.pre_sampling_sequence(individual_v, sequence_output_v_with_t, sequence_output_v_with_pv,
                                                           modality='v')  # 1024
            sequence_output_t = self.pre_sampling_sequence(individual_txt, sequence_output_t_with_v, sequence_output_t_with_pv,
                                                           modality='t')  # 768
            sequence_output_pv = self.pre_sampling_sequence(individual_pv, sequence_output_pv_with_v,
                                                            sequence_output_pv_with_t, modality='pv')  # 768
        elif self.if_pre_sampling == 0:  # mean
            sequence_output_v = (individual_v + sequence_output_v_with_t + sequence_output_v_with_pv) / 3
            sequence_output_t = (individual_txt + sequence_output_t_with_v + sequence_output_t_with_pv) / 3
            sequence_output_pv = (individual_pv + sequence_output_pv_with_v + sequence_output_pv_with_t) / 3
        elif self.if_pre_sampling == 2:  # soft
            sequence_output_v = self.pre_sampling_sequence_soft(individual_v, sequence_output_v_with_t,
                                                                sequence_output_v_with_pv, modality='v')  # 1024
            sequence_output_t = self.pre_sampling_sequence_soft(individual_txt, sequence_output_t_with_v,
                                                                sequence_output_t_with_pv, modality='t')  # 768
            sequence_output_pv = self.pre_sampling_sequence_soft(individual_pv, sequence_output_pv_with_v,
                                                                 sequence_output_pv_with_t, modality='pv')  # 768
        else:  # no fusoin
            sequence_output_v = (sequence_output_v_with_t + sequence_output_v_with_pv) / 2
            sequence_output_t = (sequence_output_t_with_v + sequence_output_t_with_pv) / 2
            sequence_output_pv = (sequence_output_pv_with_v + sequence_output_pv_with_t) / 2

        if self.use_image:
            pooled_output_v = self.map_bi_to_individual(torch.mean(sequence_output_v[:, 1:, :], dim=1))  # 1024-768
        else:
            pooled_output_v = None
        pooled_output_t = torch.mean(sequence_output_t[:, 1:, :], dim=1)  # 768
        pooled_output_pv = torch.mean(sequence_output_pv[:, 1:, :], dim=1)  # 768

        return sequence_output_v, sequence_output_t, sequence_output_pv, pooled_output_v, pooled_output_t, pooled_output_pv

    def structure_aggregator(self, c_initial, sequence_output_pv, index_p, index_v):
        ''' Compute 3 values:
            (1) initial entity embedding = pooled image embedding + pooled title embedding + pooled knowledge graph embedding
            (2) final entity embedding = initial entity embedding + attention-weighted triplets embeddings, aka Structure Aggregation Module
            (3) Link Prediction Modeling (LPM) loss

        :param pooled_output_v: pooled image embedding
        :param pooled_output_t: pooled text embedding
        :param pooled_output_pv: pooled knowledge graph embedding
        :param sequence_output_pv: sequence knowledge graph embeddings (for LPM loss)
        :param index_p: index of knowledge graph sequence
        :param index_v: index of image

        :return:
            c_initial: initial entity embedding
            c_final: final entity embedding
            loss_struct: LPM loss
        '''
        # the number of attention heads of structure aggregator in paper is 8, here is 1.
        # We found using 1 attetion head has similar performance as using 8 heads, and it's more efficient.

        #--------  structure aggregate module ------------
        property_vecs = []
        value_vecs = []
        for i in range(sequence_output_pv.shape[0]):# item
            property_vecs.append([])
            value_vecs.append([])
            for j in range(index_p.shape[1]):# p
                if index_p[i, j, 0] == 0:
                    break
                p = torch.mean(sequence_output_pv[i, :, :].index_select(dim=0, index=index_p[i, j, ]), dim=0)##[768] , keepdim=True
                v = torch.mean(sequence_output_pv[i, :, :].index_select(dim=0, index=index_v[i, j, :]), dim=0)
                property_vecs[i].append(p)
                value_vecs[i].append(v)
                if j == 0:
                    t = torch.unsqueeze(self.struc_w1(torch.cat((c_initial[i], p, v), dim=0)), 0)
                else:
                    t = torch.cat((t, torch.unsqueeze(self.struc_w1(torch.cat((c_initial[i], p, v), dim=0)), 0)), dim=0)

            try:
                b = self.struc_w2(F.leaky_relu(t))
            except:
                t = torch.unsqueeze(c_initial[i], 0)
                b = self.struc_w2(F.leaky_relu(t))

            # attention
            atten = F.softmax(b, dim=0)

            if i == 0:
                c_final = torch.unsqueeze(c_initial[i] + self.struc_w3(torch.sum(atten*t, dim=0)), 0)
                # c_final_neg = torch.unsqueeze(c_initial[i+1] + self.struc_w3(torch.sum(atten*t, dim=0)), 0)#错位构造负样本
            else:
                c_final = torch.cat((c_final, torch.unsqueeze(c_initial[i] + self.struc_w3(torch.sum(atten*t, dim=0)), 0)), dim=0)
                # c_final_neg = torch.cat((
                #     c_final_neg, torch.unsqueeze(c_initial[(i+1) % sequence_output_pv.shape[0]] + self.struc_w3(torch.sum(atten*t, dim=0)), 0)), dim=0)

        return c_final #, loss_struc

    def bert_tri(self,
                 input_txt,  # input_ids
                 input_imgs,  # image_feat
                 image_loc,  #
                 token_type_ids=None,
                 attention_mask=None,
                 image_attention_mask=None,
                 input_txt_pv=None,
                 token_type_ids_pv=None,
                 attention_mask_pv=None,
                 co_attention_mask=None,
                 task_ids=None,
                 output_all_encoded_layers=False,
                 output_all_attention_masks=False,):
        if attention_mask is None:
            attention_mask = torch.ones_like(input_txt)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_txt)
        if attention_mask_pv is None:
            attention_mask_pv = torch.ones_like(input_txt_pv)
        if token_type_ids_pv is None:
            token_type_ids_pv = torch.zeros_like(input_txt_pv)
        if image_attention_mask is None and self.use_image:
            image_attention_mask = torch.ones(
                input_imgs.size(0), input_imgs.size(1)
            ).type_as(input_txt)

        if self.task_specific_tokens:
            # extend the mask
            mask_tokens = input_txt.new().resize_(input_txt.size(0), 1).fill_(1)
            attention_mask = torch.cat([mask_tokens, attention_mask], dim=1)
            # add for PV
            mask_tokens_pv = input_txt_pv.new().resize_(input_txt_pv.size(0), 1).fill_(1)
            attention_mask_pv = torch.cat([mask_tokens_pv, attention_mask_pv], dim=1)

        # We create a 3D attention mask from a 2D tensor mask.
        # Sizes are [batch_size, 1, 1, to_seq_length]
        # So we can broadcast to [batch_size, num_heads, from_seq_length, to_seq_length]
        # this attention mask is more simple than the triangular masking of causal attention
        # used in OpenAI GPT, we just need to prepare the broadcast dimension here.
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask_pv = attention_mask_pv.unsqueeze(1).unsqueeze(2)


        extended_attention_mask2 = attention_mask.unsqueeze(2)
        extended_attention_mask2_pv = attention_mask_pv.unsqueeze(2)
        # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
        # masked positions, this operation will create a tensor which is 0.0 for
        # positions we want to attend and -10000.0 for masked positions.
        # Since we are adding it to the raw scores before the softmax, this is
        # effectively the same as removing these entirely.
        extended_attention_mask = extended_attention_mask.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        extended_attention_mask2 = extended_attention_mask2.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        # add for PV
        extended_attention_mask_pv = extended_attention_mask_pv.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility
        extended_attention_mask_pv = (1.0 - extended_attention_mask_pv) * -10000.0
        extended_attention_mask2_pv = extended_attention_mask2_pv.to(
            dtype=next(self.parameters()).dtype
        )  # fp16 compatibility

        if self.use_image:
            extended_image_attention_mask = image_attention_mask.unsqueeze(1).unsqueeze(2)
            extended_image_attention_mask = extended_image_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            extended_image_attention_mask = (1.0 - extended_image_attention_mask) * -10000.0
            if co_attention_mask is None:
                co_attention_mask = torch.zeros(
                    input_txt.size(0), input_imgs.size(1), input_txt.size(1)
                ).type_as(extended_image_attention_mask)
            extended_co_attention_mask = co_attention_mask.unsqueeze(1)  # 属于图片
            # extended_co_attention_mask = co_attention_mask.unsqueeze(-1)
            extended_co_attention_mask = extended_co_attention_mask * 5.0
            extended_co_attention_mask = extended_co_attention_mask.to(
                dtype=next(self.parameters()).dtype
            )  # fp16 compatibility
            v_embedding_output = self.v_embeddings(input_imgs, image_loc)
        else:
            extended_image_attention_mask = None
            extended_co_attention_mask = None
            v_embedding_output = None

        embedding_output = self.embeddings(input_txt, token_type_ids, task_ids)
        embedding_output_pv = self.embeddings(input_txt_pv, token_type_ids_pv, task_ids)

        (encoded_layers_t_with_v, encoded_layers_v_with_t, all_attention_mask), (
            encoded_layers_pv_with_v, encoded_layers_v_with_pv, all_attention_mask_v_pv), (
            encoded_layers_t_with_pv, encoded_layers_pv_with_t, all_attention_mask_t_pv) = self.encoder(
            embedding_output,  # text
            v_embedding_output,
            extended_attention_mask,  # text
            extended_attention_mask2,  # text
            extended_image_attention_mask,
            extended_co_attention_mask,

            embedding_output_pv,
            extended_attention_mask_pv,
            extended_attention_mask2_pv,

            output_all_encoded_layers=output_all_encoded_layers,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_t_with_pv = encoded_layers_t_with_pv[-1]
        sequence_output_pv_with_t = encoded_layers_pv_with_t[-1]
        pooled_output_t_with_pv = self.t_pooler(sequence_output_t_with_pv)
        pooled_output_pv_with_t = self.t_pooler(sequence_output_pv_with_t)

        if self.use_image:
            sequence_output_t_with_v = encoded_layers_t_with_v[-1]
            pooled_output_t_with_v = self.t_pooler(sequence_output_t_with_v)
            sequence_output_v_with_t = encoded_layers_v_with_t[-1]
            pooled_output_v_with_t = self.v_pooler(sequence_output_v_with_t)
            sequence_output_pv_with_v = encoded_layers_pv_with_v[-1]
            sequence_output_v_with_pv = encoded_layers_v_with_pv[-1]
            pooled_output_pv_with_v = self.t_pooler(sequence_output_pv_with_v)
            pooled_output_v_with_pv = self.v_pooler(sequence_output_v_with_pv)
        else:
            pooled_output_t_with_v = None
            pooled_output_v_with_t = None
            pooled_output_pv_with_v = None
            pooled_output_v_with_pv = None

        if not output_all_encoded_layers:
            encoded_layers_t_with_pv = encoded_layers_t_with_pv[-1]
            encoded_layers_pv_with_t = encoded_layers_pv_with_t[-1]
            if self.use_image:
                encoded_layers_t_with_v = encoded_layers_t_with_v[-1]
                encoded_layers_v_with_t = encoded_layers_v_with_t[-1]
                encoded_layers_pv_with_v = encoded_layers_pv_with_v[-1]
                encoded_layers_v_with_pv = encoded_layers_v_with_pv[-1]
            else:
                encoded_layers_t_with_v = None
                encoded_layers_v_with_t = None
                encoded_layers_pv_with_v = None
                encoded_layers_v_with_pv = None

        return (
            (encoded_layers_t_with_v,
             encoded_layers_v_with_t,
             pooled_output_t_with_v,
             pooled_output_v_with_t,
             all_attention_mask),
            (encoded_layers_pv_with_v,
             encoded_layers_v_with_pv,
             pooled_output_pv_with_v,
             pooled_output_v_with_pv,
             all_attention_mask_v_pv),
            (encoded_layers_t_with_pv,
             encoded_layers_pv_with_t,
             pooled_output_t_with_pv,
             pooled_output_pv_with_t,
             all_attention_mask_t_pv),
            (embedding_output,
             embedding_output_pv,
             v_embedding_output)
        )

    def item_embedding(
            self,
            input_ids,
            image_feat,
            image_loc,
            token_type_ids=None,  # segnents
            attention_mask=None,  # input_mask
            image_attention_mask=None,  # image_mask
            output_all_attention_masks=False,
            # pv
            input_ids_pv=None,
            token_type_ids_pv=None,  # segnents
            attention_mask_pv=None,
            index_p=None,
            index_v=None
    ):
        (sequence_output_t_with_v, sequence_output_v_with_t, pooled_output_t_with_v, pooled_output_v_with_t, all_attention_mask_t_v), (
            sequence_output_pv_with_v, sequence_output_v_with_pv, pooled_output_pv_with_v, pooled_output_v_with_pv,
            all_attention_mask_v_pv), (
            sequence_output_t_with_pv, sequence_output_pv_with_t, pooled_output_t_with_pv, pooled_output_pv_with_t,
            all_attention_mask_t_pv), (individual_txt, individual_pv, individual_v) = self.bert_tri(
            input_ids,
            image_feat,
            image_loc,
            token_type_ids,
            attention_mask,
            image_attention_mask,
            input_ids_pv,
            token_type_ids_pv,
            attention_mask_pv,
            output_all_encoded_layers=False,
            output_all_attention_masks=output_all_attention_masks,
        )

        sequence_output_v, sequence_output_t, sequence_output_pv, pooled_output_v, pooled_output_t, pooled_output_pv = self.get_sequence_pooled_output_final(
            sequence_output_t_with_v, sequence_output_v_with_t,
            sequence_output_pv_with_v, sequence_output_v_with_pv,
            sequence_output_t_with_pv, sequence_output_pv_with_t,
            individual_txt, individual_pv, individual_v)

        # initial item embedding
        if pooled_output_v is not None:
            c_initial = (pooled_output_v + pooled_output_t + pooled_output_pv) / 3 #[batch_size,768]
        else:
            c_initial = (pooled_output_t + pooled_output_pv) / 2

        c_final = self.structure_aggregator(c_initial, sequence_output_pv, index_p, index_v)

        return c_initial, c_final

    def forward(
            self,
            labels,
            # item 1
            input_ids_1,
            token_type_ids_1,
            attention_mask_1,
            input_ids_pv_1,
            token_type_ids_pv_1,
            attention_mask_pv_1,
            index_p_1,
            index_v_1,
            image_feat_1,
            image_loc_1,
            image_attention_mask_1,
            # item 2
            input_ids_2,
            token_type_ids_2,
            attention_mask_2,
            input_ids_pv_2,
            token_type_ids_pv_2,
            attention_mask_pv_2,
            index_p_2,
            index_v_2,
            image_feat_2,
            image_loc_2,
            image_attention_mask_2,
            output_all_attention_masks=False
    ):
        _, item_embedding_1 = self.item_embedding(input_ids_1,
                                                  image_feat_1,
                                                  image_loc_1,
                                                  token_type_ids_1,  # segnents
                                                  attention_mask_1,  # input_mask
                                                  image_attention_mask_1,  # image_mask
                                                  output_all_attention_masks,
                                                  # pv
                                                  input_ids_pv_1,
                                                  token_type_ids_pv_1,  # segnents
                                                  attention_mask_pv_1,
                                                  index_p_1,
                                                  index_v_1)
        _, item_embedding_2 = self.item_embedding(input_ids_2,
                                                  image_feat_2,
                                                  image_loc_2,
                                                  token_type_ids_2,  # segnents
                                                  attention_mask_2,  # input_mask
                                                  image_attention_mask_2,  # image_mask
                                                  output_all_attention_masks,
                                                  # pv
                                                  input_ids_pv_2,
                                                  token_type_ids_pv_2,  # segnents
                                                  attention_mask_pv_2,
                                                  index_p_2,
                                                  index_v_2)
        probs, loss = None, None
        # use inner product as logits
        if self.loss_type == "inner":
            bs, hs = item_embedding_1.shape
            inner_products = torch.bmm(item_embedding_1.view(bs, 1, hs), item_embedding_2.view(bs, hs, 1)).reshape(-1)
            loss = self.loss_fct(inner_products, labels)
            probs = 1 / (1 + torch.exp(-inner_products))
        elif self.loss_type == "cosine":
            loss = self.loss_fct(item_embedding_1, item_embedding_2, 2*labels-1)
            probs = (self.cosine(item_embedding_1, item_embedding_1) + 1) / 2
        elif self.loss_type == "ce":
            logits = self.classifier(torch.cat((item_embedding_1, item_embedding_2), dim=1))
            probs = self.softmax(logits)
            loss = self.loss_fct(logits.view(-1, 2), labels.view(-1).to(torch.long))
            item_embedding_1 = probs[:, 0]
            item_embedding_2 = probs[:, 1]
            probs = probs[:, 1]
        else:
            logger.error("Unsupported type of loss function")

        return item_embedding_1, item_embedding_2, probs, loss


