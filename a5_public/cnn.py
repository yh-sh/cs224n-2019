#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CS224N 2018-19: Homework 5
"""
import torch.nn as nn
import torch
### YOUR CODE HERE for part 1i
class CNN(nn.Module):
    
    def __init__(self, word_embed_size,  char_embed_size = 50, kernel = 5, word_length=21):

        super(CNN, self).__init__()
        self.char_embed_size = char_embed_size
        self.word_embed_size = word_embed_size
        self.kernel = kernel
        #self.in_channel = in_channel
        #self.out_channel = out_channel
        self.word_length = word_length

        self.Conv = nn.Conv1d(self.char_embed_size, self.word_embed_size, self.kernel, stride=1)
        self.maxpooling = nn.MaxPool1d(self.word_length - self.kernel + 1)

    def forward(self, X_reshaped):
        """CNN Module

        @param X_reshape :  a torch.Tensor with shape (batch_size, char_embed_size, word_length)

        @return X_conv_out: a torch.Tensor with shape (batch_size, word_embed_size)
        """
        # x_conv will be (batch_size, word_embed_size, word_length - k + 1)

        x_conv = self.Conv(X_reshaped)
        # x_conv_out will be (batch_size, word_embed_size, 1)
        x_conv = nn.functional.relu(x_conv)
        x_conv_out = self.maxpooling(x_conv)
        x_conv_out = torch.squeeze(x_conv_out, -1)
        return x_conv_out
### END YOUR CODE

