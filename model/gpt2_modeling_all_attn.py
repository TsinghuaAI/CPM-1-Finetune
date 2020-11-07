# coding=utf-8
# Copyright (c) 2019, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""GPT-2 model."""

import torch
import torch.nn.functional as F

import mpu


def init_method_normal(std=0.02):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


class AllAttnGPT2Model(torch.nn.Module):
    """GPT-2 Language model.

    The output of the forward method are the logits (parallel or
    serial depending on the `parallel_output` flag.
    """

    def __init__(self,
                 num_layers,
                 vocab_size,
                 hidden_size,
                 num_attention_heads,
                 embedding_dropout_prob,
                 attention_dropout_prob,
                 output_dropout_prob,
                 max_sequence_length,
                 checkpoint_activations,
                 checkpoint_num_layers=1,
                 parallel_output=True):

        super(AllAttnGPT2Model, self).__init__()

        self.parallel_output = parallel_output

        init_method = init_method_normal(std=0.02)

        # Word embeddings (parallel).
        self.word_embeddings = mpu.VocabParallelEmbedding(
            vocab_size, hidden_size, init_method=init_method)

        self.cls = torch.nn.Linear(3 * hidden_size, 1)

        # Position embedding (serial).
        self.position_embeddings = torch.nn.Embedding(max_sequence_length,
                                                      hidden_size)
        # Initialize the position embeddings.
        init_method(self.position_embeddings.weight)

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

        # Transformer
        self.transformer = mpu.GPT2ParallelTransformer(num_layers,
                                                       hidden_size,
                                                       num_attention_heads,
                                                       attention_dropout_prob,
                                                       output_dropout_prob,
                                                       checkpoint_activations,
                                                       checkpoint_num_layers)

    def forward(self, input_ids, position_ids, attention_mask, cand_poses, mask_pos, sent_pos):

        # Embeddings.
        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = words_embeddings + position_embeddings

        # Dropout.
        embeddings = self.embedding_dropout(embeddings)

        # Transformer.
        transformer_output = self.transformer(embeddings, attention_mask)

        # if torch.distributed.get_rank() in [0, 1]:
        #     print(transformer_output)
        #     print(transformer_output.size())
        
        # Parallel logits.
        transformer_output_parallel = mpu.copy_to_model_parallel_region(
            transformer_output)
        
        # if torch.distributed.get_rank() in [0, 1]:
        #     print(transformer_output_parallel)
        #     print(transformer_output_parallel.size())

        bs = cand_poses.size(0)

        cand_logits = torch.gather(transformer_output_parallel, 1, cand_poses.unsqueeze(-1).expand(bs, 10, transformer_output_parallel.size(-1)))
        mask_logits = torch.gather(transformer_output_parallel, 1, mask_pos.unsqueeze(-1).unsqueeze(-1).expand(bs, 1, transformer_output_parallel.size(-1))).repeat(1, 10, 1)
        sent_logits = torch.gather(transformer_output_parallel, 1, sent_pos.unsqueeze(-1).unsqueeze(-1).expand(bs, 1, transformer_output_parallel.size(-1))).repeat(1, 10, 1)

        logits_parallel = torch.cat((cand_logits, mask_logits, sent_logits), dim=-1)

        logits_parallel = self.cls(logits_parallel).squeeze(-1)

        # if torch.distributed.get_rank() in [0, 1]:
        #     print(logits_parallel)
        #     print(logits_parallel.size())

        if self.parallel_output:
            return logits_parallel

        t = mpu.gather_from_model_parallel_region(logits_parallel)
        return t


def judge_name(name):
    if "embedding" in name:
        return False
    for i in range(28):
        if str(i) in name:
            return False
    return True

def gpt2_get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for name, module_ in module.named_modules():
        # if not judge_name(name):
        #     continue
        if isinstance(module_, (mpu.LayerNorm, torch.nn.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    print(len(weight_decay_params['params']), len(no_weight_decay_params['params']))
    return weight_decay_params, no_weight_decay_params
