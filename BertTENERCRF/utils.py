# coding=utf-8

import torch
import os
import datetime
import unicodedata
import json


class InputFeatures(object):
    def __init__(self, input_id, label_id, input_mask):
        self.input_id = input_id
        self.label_id = label_id
        self.input_mask = input_mask


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = {}
    index = 0
    with open(vocab_file, "r", encoding="utf-8") as reader:
        while True:
            token = reader.readline()
            if not token:
                break
            token = token.strip()
            vocab[token] = index
            index += 1
    return vocab


def extend_maps(word2id, tag2id, for_crf=True):
    """
    扩展词汇表和标签映射
    :param word2id: 词汇表映射
    :param tag2id: 标签映射
    :param for_crf: 是否为CRF模型添加特殊标签
    :return: 扩展后的映射
    """
    # 为词汇表添加特殊标记（如果不存在）
    special_tokens = ['[PAD]', '[UNK]', '[CLS]', '[SEP]']
    for token in special_tokens:
        if token not in word2id:
            word2id[token] = len(word2id)

    # 为CRF添加特殊标签（如果不存在）
    if for_crf:
        special_tags = ['<pad>', '<start>', '<eos>']
        for tag in special_tags:
            if tag not in tag2id:
                tag2id[tag] = len(tag2id)

    return word2id, tag2id


def prepocess_data_for_lstmcrf(word_lists, tag_lists, test=False, vocab=None, label_dic=None):
    """
    为LSTM-CRF模型预处理数据
    :param word_lists: 词汇列表文件路径或数据
    :param tag_lists: 标签列表文件路径或数据
    :param test: 是否为测试数据
    :param vocab: 词汇表
    :param label_dic: 标签字典
    :return: 处理后的数据列表
    """
    if isinstance(word_lists, str):
        # 如果输入是文件路径，使用read_corpus读取
        max_length = 128  # 可以作为参数传入
        features = read_corpus(word_lists, max_length, label_dic, vocab)

        # 转换为(input_ids, label_ids)格式的元组列表
        processed_data = []
        for feature in features:
            processed_data.append((feature.input_id, feature.label_id))

        return processed_data
    else:
        # 如果输入是数据列表，直接处理
        processed_data = []
        for words, tags in zip(word_lists, tag_lists):
            # 将词转换为ID
            word_ids = [vocab.get(word, vocab.get('[UNK]', 0)) for word in words]
            # 将标签转换为ID
            tag_ids = [label_dic.get(tag, 0) for tag in tags]

            processed_data.append((word_ids, tag_ids))

        return processed_data


def read_corpus(path, max_length, label_dic, vocab):
    """
    :param path:数据文件路径
    :param max_length: 最大长度
    :param label_dic: 标签字典
    :return:
    """
    file = open(path, encoding='utf-8')
    content = file.readlines()
    file.close()
    result = []
    for line in content:
        try:
            text, label = line.strip().split('|||')
        except:
            print("Erro in data", line)
            continue
        tokens = text.split()
        label = label.split()
        if len(tokens) > max_length - 2:
            tokens = tokens[0:(max_length - 2)]
            label = label[0:(max_length - 2)]
        tokens_f = ['[CLS]'] + tokens + ['[SEP]']
        label_f = ["<start>"] + label + ['<eos>']
        input_ids = [int(vocab[i]) if i in vocab else int(vocab['[UNK]']) for i in tokens_f]
        label_ids = [label_dic[i] for i in label_f]
        input_mask = [1] * len(input_ids)
        while len(input_ids) < max_length:
            input_ids.append(0)
            input_mask.append(0)
            label_ids.append(label_dic['<pad>'])

        if not (len(input_ids) == len(input_mask) == len(label_ids) == max_length):
            print("样本长度异常，跳过此行：", line)
            continue
        feature = InputFeatures(input_id=input_ids, input_mask=input_mask, label_id=label_ids)
        result.append(feature)
    return result


def save_model(model, epoch=None, path='result', **kwargs):
    """
    只保留最优模型
    :param model: 模型
    :param path: 保存路径
    :param kwargs: 必须包含 'best'，如果是True就保存
    :return:
    """
    if not os.path.exists(path):
        os.makedirs(path)

    if kwargs.get('best', False):
        # 只在最优的时候保存
        name = 'best_model.pt'
        full_name = os.path.join(path, name)
        torch.save(model.state_dict(), full_name)
        print('Saved best model at epoch {} successfully'.format(epoch))
        # 写一个checkpoint记录
        with open(os.path.join(path, 'checkpoint'), 'w') as file:
            file.write(name)
            print('Write to checkpoint')
    else:
        # 如果不是best，不保存
        pass


def load_model(model, path=r'result', **kwargs):
    """
    加载模型
    :param model: 模型实例
    :param path: 模型路径
    :param kwargs: 可选参数，包含name指定具体文件名
    :return: 加载权重后的模型
    """
    if kwargs.get('name', None) is None:
        checkpoint_file = os.path.join(path, 'checkpoint')
        if os.path.exists(checkpoint_file):
            with open(checkpoint_file) as file:
                content = file.read().strip()
                name = os.path.join(path, content)
        else:
            # 如果没有checkpoint文件，尝试默认名称
            name = os.path.join(path, 'best_model.pt')
    else:
        name = kwargs['name']
        if not os.path.isabs(name):  # 如果不是绝对路径
            name = os.path.join(path, name)

    # 检查文件是否存在
    if not os.path.exists(name):
        raise FileNotFoundError(f"Model file {name} not found")

    model.load_state_dict(torch.load(name, map_location=lambda storage, loc: storage))
    print('load model {} successfully'.format(name))
    return model


def save_config(config, path="result/config.json"):
    """保存训练配置"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    config_dict = {}
    for key, value in config.__dict__.items():
        if not callable(value):  # 只保存非函数属性
            config_dict[key] = value

    with open(path, 'w', encoding='utf-8') as f:
        json.dump(config_dict, f, indent=2, ensure_ascii=False)
    print(f"Config saved to {path}")


def load_config(config_class, path="result/config.json"):
    """加载训练配置"""
    if not os.path.exists(path):
        print(f"Config file {path} not found, using default config")
        return config_class()

    with open(path, 'r', encoding='utf-8') as f:
        config_dict = json.load(f)

    config = config_class()
    for key, value in config_dict.items():
        setattr(config, key, value)

    print(f"Config loaded from {path}")
    return config
