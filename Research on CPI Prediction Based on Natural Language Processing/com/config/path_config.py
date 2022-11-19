#!/usr/bin/env python
# -*- coding: utf-8 -*-

# @Date: 2021/8/13
# @Author: qing
import os


class PathConfig(object):

    def __init__(self, data_path=None, save_path=None, log_path=None):

        self.root_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

        self.data_path = data_path
        self.save_path = save_path
        self.log_path = log_path

        if not data_path:
            self.data_path = os.path.join(self.root_path, "data/")

        if not save_path:
            self.save_path = self.data_path

        if not log_path:
            self.log_path = os.path.join(self.root_path, "log/")