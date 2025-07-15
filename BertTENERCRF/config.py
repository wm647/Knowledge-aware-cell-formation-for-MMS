# coding=utf-8

class Config(object):
    def __init__(self):
        self.label_file = 'BertTENERCRF/data/tag.txt'
        self.train_file = 'BertTENERCRF/data/train.txt'
        self.dev_file = 'BertTENERCRF/data/dev.txt'
        self.test_file = 'BertTENERCRF/data/test.txt'
        self.vocab = 'BertTENERCRF/data/bert/vocab.txt'
        self.bert_path = 'BertTENERCRF/data/bert'

        '''
            运行main的时候用下面的，外部调用predict时候用上面的
        '''
        self.label_file = './data/tag.txt'
        self.train_file = './data/train.txt'
        self.dev_file = './data/dev.txt'
        self.test_file = './data/test.txt'
        self.vocab = './data/bert/vocab.txt'
        self.bert_path = './data/bert'

        self.max_length = 256
        self.use_cuda = True
        self.gpu = 0
        self.batch_size = 128
        self.bert_embedding = 768

        # ===== TENER相关参数 =====
        # 统一使用transformer命名，避免混淆
        self.transformer_hidden = 256  # Transformer隐藏层维度
        self.transformer_layers = 6  # Transformer层数
        self.num_heads = 8  # 注意力头数
        self.feedforward_dim = 4*self.transformer_hidden  # 前馈网络维度

        # 保持向后兼容的别名
        self.rnn_hidden = self.transformer_hidden  # 向后兼容
        self.rnn_layer = self.transformer_layers  # 向后兼容

        self.dropout1 = 0.5
        self.dropout_ratio = 0.5
        self.lr = 1e-4
        self.lr_decay = 0
        self.weight_decay = 1e-4
        self.checkpoint = 'result/'
        self.optim = 'Adam'
        self.load_model = False
        self.load_path = None
        self.base_epoch = 600

        # 新增参数
        self.gradient_clip = 1.0
        self.scheduler = None
        # self.scheduler = 'cosine'

    def validate_params(self):
        """验证Transformer相关参数的合理性"""
        # 检查hidden_dim是否能被num_heads整除
        if self.transformer_hidden % self.num_heads != 0:
            print(
                f"Warning: transformer_hidden ({self.transformer_hidden}) should be divisible by num_heads ({self.num_heads})")
            # 自动调整num_heads
            for heads in [8, 4, 2, 1]:
                if self.transformer_hidden % heads == 0:
                    self.num_heads = heads
                    print(f"Auto-adjusted num_heads to {heads}")
                    break

        # 建议的配置组合
        if self.transformer_hidden == 256:
            recommended_heads = [8, 4, 2, 1]
        elif self.transformer_hidden == 512:
            recommended_heads = [8, 16, 4]
        elif self.transformer_hidden == 768:
            recommended_heads = [12, 8, 6, 4]
        else:
            recommended_heads = [1]

        if self.num_heads not in recommended_heads:
            print(f"Recommended num_heads for hidden_dim {self.transformer_hidden}: {recommended_heads}")

        # 同步向后兼容的参数
        self.rnn_hidden = self.transformer_hidden
        self.rnn_layer = self.transformer_layers

    def update(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        # 更新后重新验证参数
        self.validate_params()

    def __str__(self):
        return '\n'.join(['%s:%s' % item for item in self.__dict__.items()])


if __name__ == '__main__':
    con = Config()
    con.update(gpu=8)
    print(con.gpu)
    print(con)
