#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch
### YOUR CODE HERE for part 1h
class Highway(nn.Module):
    def __init__(self, word_embed_size: int):
        super(Highway, self).__init__()
        self.embed_size = word_embed_size

        self.proj_Linear = nn.Linear(self.embed_size, self.embed_size)
        self.gate_Linear = nn.Linear(self.embed_size, self.embed_size)
    def forward(self, Conv_out : torch.Tensor):
        """Model highway
        
        @param Conv_out: a Tensor with size(batch_size, word_embed)

        @return Word_embed: a tensor with size(batch_size, word_embed)
        """
        X_proj = nn.functional.relu(self.proj_Linear(Conv_out))
        X_gate = torch.sigmoid(self.gate_Linear(Conv_out))
        Word_embed = X_gate.mul(X_proj) + (1 - X_gate).mul(Conv_out)
        return Word_embed
### END YOUR CODE 



