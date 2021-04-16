import copy
import torch
import torch.nn as nn
import torch.nn.functional as F

import mpu

from .configuration_enc_dec import EncDecConfig


def init_method_normal(std):
    """Init method based on normal distribution.

    This is only used for embeddings. The transformer has its
    own initializer.
    """
    def init_(tensor):
        return torch.nn.init.normal_(tensor, mean=0.0, std=std)
    return init_


class EncDecModel(nn.Module):
    
    def __init__(
        self,
        config: EncDecConfig,
        parallel_output=True,
        checkpoint_activations=False,
        checkpoint_num_layers=1):
        
        super(EncDecModel, self).__init__()
        if config.vocab_size is None:
            raise RuntimeError("Should set vocab size")
        self.enc_config = copy.deepcopy(config)
        self.dec_config = copy.deepcopy(config)

        self.parallel_output = parallel_output

        init_method = init_method_normal(std=config.init_method_std) # NOTE: good?

        self.word_embeds = mpu.VocabParallelEmbedding(config.vocab_size, config.d_model, init_method=init_method)

        self.lm_head = mpu.VocabParallelEmbedding(config.vocab_size, config.d_model, init_method=init_method)

        self.encoder = mpu.ParallelTransformer(self.enc_config, word_embeds=self.word_embeds, is_decoder=False,
                                               checkpoint_activations=checkpoint_activations, checkpoint_num_layers=checkpoint_num_layers)
        self.decoder = mpu.ParallelTransformer(self.dec_config, word_embeds=self.word_embeds, is_decoder=True,
                                               checkpoint_activations=checkpoint_activations, checkpoint_num_layers=checkpoint_num_layers)

    def forward(
        self, 
        enc_input_ids=None,
        enc_position_ids=None,
        enc_attention_mask=None,
        dec_input_ids=None,
        dec_position_ids=None,
        dec_attention_mask=None,
        cross_attention_mask=None,):

        enc_outputs = self.encoder(
            input_ids=enc_input_ids,
            position_ids=enc_position_ids,
            attention_mask=enc_attention_mask,
        )

        enc_hidden_states = enc_outputs["last_hidden_state"]

        dec_outputs = self.decoder(
            input_ids=dec_input_ids,
            position_ids=dec_position_ids,
            attention_mask=dec_attention_mask,
            cross_attention_mask=cross_attention_mask,
            enc_hidden_states=enc_hidden_states
        )

        last_hidden_state_parallel = mpu.copy_to_model_parallel_region(dec_outputs["last_hidden_state"])
        logits_parallel = F.linear(last_hidden_state_parallel, self.lm_head.weight)

        if self.parallel_output:
            lm_logits = logits_parallel
        else:
            lm_logits = mpu.gather_from_model_parallel_region(logits_parallel)

        outputs = {
            "lm_logits": lm_logits,
            "last_hidden_state": dec_outputs["last_hidden_state"],
            "past_key_values": dec_outputs["past_key_values"],
            "decoder_hidden_states": dec_outputs["hidden_states"],
            "decoder_attentions": dec_outputs["attentions"],
            "cross_attentions": dec_outputs["cross_attentions"],
            "encoder_last_hidden_state": enc_outputs["last_hidden_state"],
            "encoder_hidden_states": enc_outputs["hidden_states"],
            "encoder_attentions": enc_outputs["attentions"],
        }

        return outputs


def enc_dec_get_params_for_weight_decay_optimization(module):

    weight_decay_params = {'params': []}
    no_weight_decay_params = {'params': [], 'weight_decay': 0.0}
    for module_ in module.modules():
        if isinstance(module_, (mpu.LayerNorm, nn.LayerNorm, mpu.transformer_enc_dec.LayerNorm)):
            no_weight_decay_params['params'].extend(
                [p for p in list(module_._parameters.values())
                 if p is not None])
        else:
            weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n != 'bias'])
            no_weight_decay_params['params'].extend(
                [p for n, p in list(module_._parameters.items())
                 if p is not None and n == 'bias'])

    return weight_decay_params, no_weight_decay_params


# def enc_dec_get_params_for_weight_decay_optimization(module):

#     lr_params = {'params': []}
#     no_lr_params = {'params': [], 'lr': 0.0}
#     for module_ in module.modules():
#         if isinstance(module_, (nn.Embedding, mpu.VocabParallelEmbedding)):
#             lr_params['params'].extend(
#                 [p for n, p in list(module_._parameters.items())
#                  if p is not None])
#         else:
#             no_lr_params['params'].extend(
#                 [p for n, p in list(module_._parameters.items())
#                  if p is not None])

    # return lr_params, no_lr_params
    return (lr_params, )
