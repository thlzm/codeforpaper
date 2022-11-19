#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/9/24
# @Author: qing
import torch

from megatron.module import MegatronModule


class TopEmbedding(MegatronModule):
    """Language model embeddings.

    Arguments:
        hidden_size: hidden size
        vocab_size: vocabulary size
        max_sequence_length: maximum size of sequence. This
                             is used for positional embedding
        embedding_dropout_prob: dropout probability for embeddings
        init_method: weight initialization method
        num_tokentypes: size of the token-type embeddings. 0 value
                        will ignore this embedding
    """

    def __init__(self,
                 hidden_size,
                 vocab_size,
                 max_sequence_length,
                 embedding_dropout_prob,
                 init_method,
                 num_tokentypes=0,
                 top_query_switch=True,
                 task_switch=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.init_method = init_method
        self.num_tokentypes = num_tokentypes
        self.top_query_switch=top_query_switch
        self.task_switch=task_switch

        # Top query position embedding (serial).
        if self.top_query_switch:
            self.top_query_embeddings = torch.nn.Embedding(max_sequence_length, self.hidden_size)
            self._top_query_embeddings_key = 'top_query_embeddings'
            # Initialize the top query position embeddings.
            self.init_method(self.top_query_embeddings.weight)
        else:
            self.top_query_embeddings = None

        # Token type embedding.
        if self.task_switch and self.num_tokentypes > 0:
            self.tokentype_embeddings = torch.nn.Embedding(self.num_tokentypes, self.hidden_size)
            self._tokentype_embeddings_key = 'tasktype_embeddings'
            # Initialize the token-type embeddings.
            self.init_method(self.tokentype_embeddings.weight)
        else:
            self.tokentype_embeddings = None

        # Embeddings dropout
        self.embedding_dropout = torch.nn.Dropout(embedding_dropout_prob)

    def add_tokentype_embeddings(self, num_tokentypes):
        """Add token-type embedding. This function is provided so we can add
        token-type embeddings in case the pretrained model does not have it.
        This allows us to load the model normally and then add this embedding.
        """
        if self.tokentype_embeddings is not None:
            raise Exception('tokentype embeddings is already initialized')
        if torch.distributed.get_rank() == 0:
            print('adding embedding for {} tokentypes'.format(num_tokentypes), flush=True)
        self.num_tokentypes = num_tokentypes
        self.tokentype_embeddings = torch.nn.Embedding(num_tokentypes,
                                                       self.hidden_size)
        # Initialize the token-type embeddings.
        self.init_method(self.tokentype_embeddings.weight)

    def forward(self, position_ids, tasktype_ids):
        # Embeddings.
        embeddings = None
        if self.task_switch and self.top_query_switch:
            embeddings = self.tokentype_embeddings(tasktype_ids)
            embeddings = embeddings + self.top_query_embeddings(position_ids)
        elif self.task_switch:
            embeddings = self.tokentype_embeddings(tasktype_ids)
        elif self.top_query_switch:
            embeddings = self.top_query_embeddings(position_ids)
        else:
            embeddings = embeddings

        if embeddings != None:
            embeddings = self.embedding_dropout(embeddings)

        return embeddings

    def state_dict_for_save_checkpoint(self, destination=None, prefix='', keep_vars=False):
        """For easy load."""

        state_dict_ = {}
        if self.top_query_switch:
            state_dict_[self._top_query_embeddings_key] = self.top_query_embeddings.state_dict(destination, prefix, keep_vars)

        if self.task_switch and  self.num_tokentypes > 0:
            state_dict_[self._tokentype_embeddings_key] = self.tokentype_embeddings.state_dict(destination, prefix, keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""

        # Position embedding.
        if self._top_query_embeddings_key in state_dict:
            state_dict_ = state_dict[self._top_query_embeddings_key]
        else:
            # for backward compatibility.
            state_dict_ = {}
            for key in state_dict.keys():
                if 'top_query_embeddings' in key:
                    state_dict_[key.split('top_query_embeddings.')[1]] = state_dict[key]

        if self.top_query_switch:
            self.top_query_embeddings.load_state_dict(state_dict_, strict=strict)

        # Tokentype embedding.
        if self.task_switch and self.num_tokentypes > 0:
            state_dict_ = {}
            if self._tokentype_embeddings_key in state_dict:
                state_dict_ = state_dict[self._tokentype_embeddings_key]
            else:
                # for backward compatibility.
                for key in state_dict.keys():
                    if 'tasktype_embeddings' in key:
                        state_dict_[key.split('tasktype_embeddings.')[1]] = state_dict[key]
            if len(state_dict_.keys()) > 0:
                self.tokentype_embeddings.load_state_dict(state_dict_, strict=strict)
            else:
                print('***WARNING*** expected tasktype embeddings in the checkpoint but could not find it', flush=True)