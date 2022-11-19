#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/13
# @Author: qing

def add_custom_args(parser):
    group = parser.add_argument_group(title='custom_args')
    group.add_argument('--gradient-accumulation-steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    return parser