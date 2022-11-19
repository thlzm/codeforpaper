#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/9/24
# @Author: qing
import torch

from megatron import mpu
from megatron import get_args
from megatron.model import GPT2Model
from megatron.module import MegatronModule
from megatron.model.transformer import ParallelTransformer
from megatron.model.gpt2_model import gpt2_attention_mask_func
from megatron.model.language_model import parallel_lm_logits, Embedding, Pooler
from megatron.model.utils import init_method_normal, scaled_init_method_normal

from com.utils.mpu_util import mp_gather
from model.pangu_evolution.model.top_layer import TopEmbedding


class TransformerLanguageModelBase(MegatronModule):
    """Transformer language model.

    Arguments:
        transformer_hparams: transformer hyperparameters
        attention_mask_func: a function that takes `unmaksed-attention-scores`
            with size [b, np, s, s] and an `attention-mask` and will apply
            the masking. The function should return a masked score of the
            same size [b, np, s, s].
          masked-attention-scores = attention_mask_func(
                                     unmaksed-attention-scores, attention-mask)
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 attention_mask_func,
                 init_method,
                 output_layer_init_method,
                 num_tokentypes=0,
                 add_pooler=False):
        super(TransformerLanguageModelBase, self).__init__()
        args = get_args()

        self.hidden_size = args.hidden_size
        self.num_tokentypes = num_tokentypes
        self.init_method = init_method
        self.add_pooler = add_pooler

        # Embeddings
        self.embedding = Embedding(self.hidden_size,
                                   args.padded_vocab_size,
                                   args.max_position_embeddings,
                                   args.hidden_dropout,
                                   self.init_method,
                                   self.num_tokentypes)
        self._embedding_key = 'embedding'

        self._topQueryEmbedding_key = 'topQueryEmbedding'

        # Transformer
        self.transformer = ParallelTransformer(attention_mask_func, self.init_method, output_layer_init_method)
        self._transformer_key = 'transformer'

        # Pooler
        if self.add_pooler:
            self.pooler = Pooler(self.hidden_size, self.init_method)
            self._pooler_key = 'pooler'

    def forward(self, input_ids, position_ids, attention_mask, tokentype_ids=None, layer_past=None, get_key_value=False, pooling_sequence_index=0):
        raise NotImplementedError("TransformerLanguageModelBase has to implement forward method.")

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""
        state_dict_ = {}
        state_dict_[self._embedding_key] = self.embedding.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._transformer_key] = self.transformer.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        if self.add_pooler:
            state_dict_[self._pooler_key] = self.pooler.state_dict_for_save_checkpoint(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Embedding.
        if self._embedding_key in state_dict:
            state_dict_ = state_dict[self._embedding_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if '_embeddings' in key:
                    state_dict_[key] = state_dict[key]
        self.embedding.load_state_dict(state_dict_, strict=strict)

        # Transformer.
        if self._transformer_key in state_dict:
            state_dict_ = state_dict[self._transformer_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'transformer.' in key:
                    state_dict_[key.split('transformer.')[1]] = state_dict[key]
        self.transformer.load_state_dict(state_dict_, strict=strict)

        # Pooler.
        if self.add_pooler:
            assert 'pooler' in state_dict, 'could not find data for pooler in the checkpoint'
            self.pooler.load_state_dict(state_dict[self._pooler_key], strict=strict)


class TransformerLanguageModelEnhance(TransformerLanguageModelBase):

    def __init__(self,
                 attention_mask_func,
                 init_method,
                 output_layer_init_method,
                 num_tokentypes=0,
                 add_pooler=False,
                 num_tasktypes=0):
        super().__init__(attention_mask_func, init_method, output_layer_init_method, num_tokentypes, add_pooler)

        args = get_args()
        self.num_tasktypes = num_tasktypes

        # top embeddings
        self.top_embedding = TopEmbedding(self.hidden_size,
                                   args.padded_vocab_size,
                                   args.max_position_embeddings,
                                   args.hidden_dropout,
                                   self.init_method,
                                   self.num_tasktypes)
        self._top_embedding_key = 'task_embedding'


    def forward(self, inputs, position_ids, attention_mask, tokentype_ids=None, layer_past=None, get_key_value=False, pooling_sequence_index=0):
        input_ids, task_ids = inputs

        embedding_output = self.embedding(input_ids, position_ids, tokentype_ids=tokentype_ids)
        top_embedding_out = self.top_embedding(position_ids, tasktype_ids=task_ids)

        transformer_output = self.transformer(embedding_output,
                                              top_embedding_out,
                                              attention_mask,
                                              layer_past=layer_past,
                                              get_key_value=get_key_value)

        if self.add_pooler:
            pooled_output = self.pooler(transformer_output, pooling_sequence_index)
            return transformer_output, pooled_output

        return transformer_output


    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""
        state_dict_ = super().state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        state_dict_[self._top_embedding_key] = self.top_embedding.state_dict_for_save_checkpoint(destination, prefix, keep_vars)
        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        super().load_state_dict(state_dict, strict)

        if self._top_embedding_key in state_dict:
            state_dict_ = state_dict[self._top_embedding_key]
        elif self._topQueryEmbedding_key in state_dict:
            state_dict_ = state_dict[self._topQueryEmbedding_key]
        else:
            state_dict_ = {}
            for key in state_dict.keys():
                if '_embeddings' in key:
                    state_dict_[key] = state_dict[key]
        self.top_embedding.load_state_dict(state_dict_, strict=strict)


def get_enhance_lm(attention_mask_func, num_tokentypes, add_pooler, num_tasktypes, init_method=None, scaled_init_method=None):
    """Build language model and return along with the key to save."""
    args = get_args()

    if init_method is None:
        init_method = init_method_normal(args.init_method_std)

    if scaled_init_method is None:
        scaled_init_method = scaled_init_method_normal(args.init_method_std, args.num_layers)

    enhance_lm = TransformerLanguageModelEnhance(
        attention_mask_func=attention_mask_func,
        init_method=init_method,
        output_layer_init_method=scaled_init_method,
        num_tokentypes=num_tokentypes,
        add_pooler=add_pooler,
        num_tasktypes=num_tasktypes)

    # key used for checkpoints.
    language_model_key = 'language_model'

    return enhance_lm, language_model_key


class PanguModelEnhance(GPT2Model):

    def __init__(self, num_tasktypes=0, num_tokentypes=0, parallel_output=True):
        super().__init__(num_tokentypes=num_tokentypes, parallel_output=parallel_output, should_get_lm=False)

        args = get_args()

        self.language_model, self._language_model_key = get_enhance_lm(
            attention_mask_func=gpt2_attention_mask_func,
            num_tokentypes=num_tokentypes,
            add_pooler=False,
            num_tasktypes=num_tasktypes,
            init_method=init_method_normal(args.init_method_std),
            scaled_init_method=scaled_init_method_normal(args.init_method_std, args.num_layers))

    def forward(self, inputs, position_ids, attention_mask, labels=None, tokentype_ids=None, layer_past=None,
                get_key_value=False, forward_method_parallel_output=None, parallel_logits=True):

        lm_output = self.language_model(inputs,
                                        position_ids,
                                        attention_mask,
                                        tokentype_ids=tokentype_ids,
                                        layer_past=layer_past,
                                        get_key_value=get_key_value)

        if get_key_value:
            lm_output, presents = lm_output

        lm_output = torch.add(lm_output, 0)

        parallel_output = self.parallel_output
        if forward_method_parallel_output is not None:
            parallel_output = forward_method_parallel_output

        output = parallel_lm_logits(lm_output, self.language_model.embedding.word_embeddings.weight, parallel_output)
        # print(f"output.shape: {output.shape}")

        if get_key_value:
            output = [output, presents]

        if labels is None:
            return output
        else:
            if self.fp16_lm_cross_entropy:
                assert output.dtype == torch.half
                loss = mpu.vocab_parallel_cross_entropy(output, labels)
            else:
                loss = mpu.vocab_parallel_cross_entropy(output.float(), labels)

            if not parallel_logits:
                output_gather = mp_gather(output)
            else:
                output_gather = output

            return loss, output_gather