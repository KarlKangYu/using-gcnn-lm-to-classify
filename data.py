from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import numpy as np
import codecs

def read_lm_data(pos_data_path, neg_data_path,num_steps):
    lm_in_ids_list, lm_out_ids_list, y = [], [], []

    with codecs.open(pos_data_path, "r") as f1:
        with codecs.open(neg_data_path, 'r') as f2:
            for line in f1.readlines():
                lm_in, lm_out = line.strip().split("#")#按输入输出拆开
                lm_in_ids = lm_in.split()[:num_steps]#按每个单词拆开，最多有numsteps个单词,防止有的句子太长
                lm_out_ids = lm_out.split()[1:num_steps + 1]
                ##################
                lm_in_ids = lm_in_ids + [0] * max(num_steps - len(lm_in_ids), 0)
                lm_out_ids = lm_out_ids + [0] * max(num_steps - len(lm_out_ids), 0)
                ##################
                lm_in_ids_list.append(lm_in_ids)
                lm_out_ids_list.append(lm_out_ids)
                y.append([0, 1])

            for line in f2.readlines():
                lm_in, lm_out = line.strip().split("#")#按输入输出拆开
                lm_in_ids = lm_in.split()[:num_steps]#按每个单词拆开，最多有numsteps个单词,防止有的句子太长
                lm_out_ids = lm_out.split()[1:num_steps + 1]
                ##################
                lm_in_ids = lm_in_ids + [0] * max(num_steps - len(lm_in_ids), 0)
                lm_out_ids = lm_out_ids + [0] * max(num_steps - len(lm_out_ids), 0)
                ##################
                lm_in_ids_list.append(lm_in_ids)
                lm_out_ids_list.append(lm_out_ids)
                y.append([1, 0])

    return lm_in_ids_list, lm_out_ids_list, y

def read_file(data_path, config, is_train=False):

    mode = "train" if is_train else "dev"

    pos_lm_data_file = os.path.join(data_path, mode + "_pos_in_ids_lm")
    neg_lm_data_file = os.path.join(data_path, mode + "_neg_in_ids_lm")

    lm_in_data, lm_out_data, y = read_lm_data(pos_lm_data_file, neg_lm_data_file,config.num_steps)

    print(mode + " data size: ", len(lm_in_data))

    return [lm_in_data, lm_out_data, y]


def data_iterator(data, config):
    lm_in_data = np.array(data[0])
    lm_out_data = np.array(data[1], dtype=np.int32)
    y = np.array(data[2])
    data_size = len(lm_in_data)
    shuffle_indices = np.random.permutation(np.arange(data_size))
    lm_in_data = lm_in_data[shuffle_indices]
    lm_out_data = lm_out_data[shuffle_indices]
    y = y[shuffle_indices]

    print("DATA SIZE:", data_size)

    num_steps = config.num_steps
    batch_size = config.batch_size

    while True:
        batches_per_epoch = data_size // batch_size + 1
        for batch_num in range(batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            lm_in_epoch = lm_in_data[start_index:end_index, : ]
            lm_out_epoch = lm_out_data[start_index:end_index, :]
            y_epoch = y[start_index:end_index, :]
            yield batches_per_epoch, lm_in_epoch, lm_out_epoch, y_epoch



