# coding=utf-8
'''
注意，该代码用的是python3.8（pytorch）环境
'''

import torch
import os
import json
from datetime import datetime
from torch.autograd import Variable
from BertTENERCRF.config import Config
from BertTENERCRF.model import BERT_TENER_CRF
from BertTENERCRF.utils import load_vocab, extend_maps, prepocess_data_for_lstmcrf, load_model, save_model
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings

# 忽略特定的UserWarning
warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors")
warnings.filterwarnings("ignore", message="Torch was not compiled with flash attention")

import time
# 记录开始时间
start_time = time.time()

# 配置保存和加载函数
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


def get_model_and_data(config):
    """获取模型和数据"""
    vocab = load_vocab(config.vocab)
    label_dic = load_vocab(config.label_file)

    # 扩展词汇表和标签映射
    vocab, label_dic = extend_maps(vocab, label_dic, for_crf=True)

    tagset_size = len(label_dic)

    # 正确传递所有参数给模型
    model = BERT_TENER_CRF(
        bert_config=config.bert_path,  # BERT模型路径
        tagset_size=tagset_size,
        embedding_dim=config.bert_embedding,  # 768
        hidden_dim=config.transformer_hidden,  # 从config获取
        rnn_layers=config.transformer_layers,  # transformer层数
        dropout_ratio=config.dropout_ratio,
        dropout1=config.dropout1,
        use_cuda=config.use_cuda,
        num_heads=config.num_heads
    )

    return model, vocab, label_dic



def train(config=Config(), early_stop_patience=10):
    """训练函数"""
    print('当前设置为:\n', config)

    # 验证配置参数
    config.validate_params()

    # 保存训练配置
    save_config(config, "result/config.json")

    model, vocab, label_dic = get_model_and_data(config)

    train_data = prepocess_data_for_lstmcrf(
        word_lists=config.train_file,
        tag_lists=config.train_file,
        test=False,
        vocab=vocab,
        label_dic=label_dic
    )

    dev_data = prepocess_data_for_lstmcrf(
        word_lists=config.dev_file,
        tag_lists=config.dev_file,
        test=False,
        vocab=vocab,
        label_dic=label_dic
    )

    train_loader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=lambda x: x)
    dev_loader = DataLoader(dev_data, batch_size=config.batch_size, shuffle=False, collate_fn=lambda x: x)

    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device)

    # 优化器设置
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # 学习率调度器
    if config.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.base_epoch)
    elif config.scheduler == 'step':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    else:
        scheduler = None

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    dev_losses = []

    for epoch in range(config.base_epoch):
        model.train()
        total_loss = 0

        train_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{config.base_epoch}")
        for batch in train_bar:
            optimizer.zero_grad()

            # 处理批次数据
            sentences = []
            tags = []
            masks = []

            max_len = max(len(item[0]) for item in batch)

            for item in batch:
                sent = item[0] + [0] * (max_len - len(item[0]))  # 填充
                tag = item[1] + [0] * (max_len - len(item[1]))  # 填充
                mask = [1] * len(item[0]) + [0] * (max_len - len(item[0]))  # 注意力掩码

                sentences.append(sent)
                tags.append(tag)
                masks.append(mask)

            # 转换为张量
            sentences_tensor = torch.tensor(sentences, dtype=torch.long).to(device)
            tags_tensor = torch.tensor(tags, dtype=torch.long).to(device)
            masks_tensor = torch.tensor(masks, dtype=torch.long).to(device)

            # 前向传播
            feats = model(sentences_tensor, attention_mask=masks_tensor)
            loss = model.loss(feats, masks_tensor, tags_tensor)

            # 反向传播
            loss.backward()

            # 梯度裁剪
            if config.gradient_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip)

            optimizer.step()

            total_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        # 验证
        model.eval()
        val_loss = 0

        with torch.no_grad():
            for batch in dev_loader:
                sentences = []
                tags = []
                masks = []

                max_len = max(len(item[0]) for item in batch)

                for item in batch:
                    sent = item[0] + [0] * (max_len - len(item[0]))
                    tag = item[1] + [0] * (max_len - len(item[1]))
                    mask = [1] * len(item[0]) + [0] * (max_len - len(item[0]))

                    sentences.append(sent)
                    tags.append(tag)
                    masks.append(mask)

                sentences_tensor = torch.tensor(sentences, dtype=torch.long).to(device)
                tags_tensor = torch.tensor(tags, dtype=torch.long).to(device)
                masks_tensor = torch.tensor(masks, dtype=torch.long).to(device)

                feats = model(sentences_tensor, attention_mask=masks_tensor)
                loss = model.loss(feats, masks_tensor, tags_tensor)
                val_loss += loss.item()

        avg_train_loss = total_loss / len(train_loader)
        avg_val_loss = val_loss / len(dev_loader)

        train_losses.append(avg_train_loss)
        dev_losses.append(avg_val_loss)

        print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {avg_val_loss:.4f}")

        # 学习率调度
        if scheduler:
            scheduler.step()

        # 早停机制
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # 保存最佳模型
            save_model(model, epoch, "result", best=True)
        else:
            patience_counter += 1
            if patience_counter >= early_stop_patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break

    print("Training completed!")
    # 绘制并保存 Loss 曲线
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(10, 6))
    plt.plot(epochs, train_losses, label='Train Loss', linewidth=2)
    plt.plot(epochs, dev_losses, label='Dev Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize = 16)
    plt.ylabel('Loss', fontsize = 16)
    plt.title('Training & Validation Loss Curve', fontsize = 16)
    plt.legend(fontsize = 16)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(datetime.now().strftime("%Y-%m-%d %H %M %S")+'loss_curve.png')
    plt.show()

def predict(text, tag_list=None, config=None):
    """预测函数"""
    if config is None:
        config = load_config(Config, "result/config.json")
        print("使用训练时的配置进行预测")

    model, vocab, label_dic = get_model_and_data(config)

    # 加载模型
    model_paths = [
        "result/best_model.pt",
        "BertTENERCRF/result/best_model.pt",
        "best_model.pt"
    ]

    # model_paths = [
    #     "result/6.4的模型.pt",
    #     "./result/best_model.pt",
    #     "best_model.pt"
    # ]

    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = load_model(model, path=os.path.dirname(model_path), name=os.path.basename(model_path))
                model_loaded = True
                print(f"成功加载模型: {model_path}")
                break
            except Exception as e:
                print(f"加载模型 {model_path} 失败: {e}")
                continue

    if not model_loaded:
        print("警告：未找到训练好的模型文件，使用未训练的模型")

    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    # 文本预处理
    tokens = ['[CLS]'] + text.strip().split() + ['[SEP]']
    print("这是tokens", tokens)
    token_ids = [vocab.get(token, vocab.get('[UNK]', 0)) for token in tokens]
    attention_mask = [1] * len(token_ids)

    tokens_tensor = torch.tensor([token_ids], dtype=torch.long).to(device)
    mask_tensor = torch.tensor([attention_mask], dtype=torch.long).to(device)

    with torch.no_grad():
        # 前向传播获取特征
        feats = model(tokens_tensor, attention_mask=mask_tensor)

        # 获取真实的tag数量（不含START和END）
        tag_size = model.crf.target_size

        print("[DEBUG] 输入 tokens:", tokens)
        print("[DEBUG] token_ids 长度:", len(token_ids))
        print("[DEBUG] feats.shape:", feats.shape)
        print("[DEBUG] tag_size (不含START/END):", tag_size)

        # CRF 解码
        _, predicted_tags_list = model.crf(feats, mask_tensor)

        # predicted_tags_list 已经是 list，直接取第一个batch的结果
        predicted_tags = predicted_tags_list[0]  # 这已经是Python list

        print("[DEBUG] predicted_tags类型:", type(predicted_tags))
        print("[DEBUG] predicted_tags长度:", len(predicted_tags))
        print("[DEBUG] predicted_tags内容:", predicted_tags)

        # 转换为标签名称
        inv_label = {v: k for k, v in label_dic.items()}
        predicted_labels = [inv_label.get(t, 'O') for t in predicted_tags]

        print("[DEBUG] 预测标签:", predicted_labels)

        # 重要修改：移除CLS和SEP对应的标签
        # 我们需要跳过第一个标签(<start>)和最后一个标签(<eos>)
        predicted_labels = predicted_labels[1:-1]  # 去除首尾特殊标签

        print("[DEBUG] 处理后的预测标签:", predicted_labels)

        # 计算损失（如果提供了真实标签）
        loss = None
        if tag_list is not None:
            # 将tag_list转换为tag_ids，包含CLS和SEP的标签
            tag_ids = []
            tag_ids.append(label_dic.get('O', 0))  # CLS标签
            for tag in tag_list:
                tag_ids.append(label_dic.get(tag, label_dic.get('O', 0)))
            tag_ids.append(label_dic.get('O', 0))  # SEP标签

            # 确保长度一致
            if len(tag_ids) != len(token_ids):
                print(f"[WARNING] 标签长度({len(tag_ids)})与token长度({len(token_ids)})不匹配")
                # 截断或填充
                if len(tag_ids) > len(token_ids):
                    tag_ids = tag_ids[:len(token_ids)]
                else:
                    tag_ids.extend([label_dic.get('O', 0)] * (len(token_ids) - len(tag_ids)))

            true_tags_tensor = torch.tensor([tag_ids], dtype=torch.long).to(device)
            loss = model.loss(feats, mask_tensor, true_tags_tensor).item()

    # 返回时需要注意长度对齐
    input_text_tokens = text.strip().split()

    # 确保predicted_labels和input_text_tokens长度一致
    if len(predicted_labels) != len(input_text_tokens):
        print(f"[WARNING] 标签长度({len(predicted_labels)})与文本长度({len(input_text_tokens)})不匹配")
        # 调整长度
        if len(predicted_labels) > len(input_text_tokens):
            predicted_labels = predicted_labels[:len(input_text_tokens)]
        else:
            predicted_labels.extend(['O'] * (len(input_text_tokens) - len(predicted_labels)))

    return input_text_tokens, predicted_labels, loss



def evaluate_on_validation_set(config=None, val_path=None, method: str = 'batch') -> float:
    """在验证集上计算损失"""
    if config is None:
        config = load_config(Config, "result/config.json")

    model, vocab, label_dic = get_model_and_data(config)

    # 加载模型
    model_paths = [
        "result/best_model.pt",
        "./result/best_model.pt",
        "best_model.pt"
    ]

    model_loaded = False
    for model_path in model_paths:
        if os.path.exists(model_path):
            try:
                model = load_model(model, path=os.path.dirname(model_path), name=os.path.basename(model_path))
                model_loaded = True
                break
            except Exception as e:
                continue

    if not model_loaded:
        raise FileNotFoundError("未找到训练好的模型文件")

    device = torch.device("cuda" if config.use_cuda and torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    if val_path is None:
        val_path = config.dev_file

    val_data = prepocess_data_for_lstmcrf(
        word_lists=val_path,
        tag_lists=val_path,
        test=False,
        vocab=vocab,
        label_dic=label_dic
    )

    val_loader = DataLoader(val_data, batch_size=config.batch_size, shuffle=False, collate_fn=lambda x: x)

    total_loss = 0

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            sentences = []
            tags = []
            masks = []

            max_len = max(len(item[0]) for item in batch)

            for item in batch:
                sent = item[0] + [0] * (max_len - len(item[0]))
                tag = item[1] + [0] * (max_len - len(item[1]))
                mask = [1] * len(item[0]) + [0] * (max_len - len(item[0]))

                sentences.append(sent)
                tags.append(tag)
                masks.append(mask)

            sentences_tensor = torch.tensor(sentences, dtype=torch.long).to(device)
            tags_tensor = torch.tensor(tags, dtype=torch.long).to(device)
            masks_tensor = torch.tensor(masks, dtype=torch.long).to(device)

            feats = model(sentences_tensor, attention_mask=masks_tensor)
            loss = model.loss(feats, masks_tensor, tags_tensor)
            total_loss += loss.item()

    avg_loss = total_loss / len(val_loader)
    print(f"验证集平均损失: {avg_loss:.4f}")
    return avg_loss


if __name__ == '__main__':
    mode = 'train'
    # mode = 'predict'
    # mode = 'eval'

    def main():
        pass

    if mode == 'train':
        # 创建TENER专用配置
        tener_config = Config()
        # tener_config.update(
        #     transformer_hidden=256,
        #     transformer_layers=4,
        #     num_heads=8,
        #     dropout_ratio=0.2,
        #     lr=1e-5,
        #     batch_size=16,
        #     gradient_clip=1.0,
        #     scheduler='cosine'
        # )
        train(config=tener_config, early_stop_patience=10)

    elif mode == 'predict':
        text = "按 整 C A G Z P T 将 G S Y Q a D 装 在 电 源 单 元 L n B 机 架 上"
        tag_list = ["O", "B-Pf", "I-Pf", "I-Pf", "I-Pf", "I-Pf", "I-Pf", "E-Pf", "O", "B-Co", "I-Co", "I-Co", "I-Co",
                    "I-Co", "E-Co", "S-Pr", "O", "B-Fm", "I-Fm", "I-Fm", "I-Fm", "I-Fm", "I-Fm", "E-Fm", "B-Co", "E-Co",
                    "O"]
        # text = "件 1 9 （ W o L 组 件 ： W L - 3 A ） 安 装 于 件 5 （ C A G 段 单 元 ） 上 ， 如 果 安 装 间 隙 大 于 0 . 2 m m 时 ， 根 据 间 歇 大 小 采 用 件 5 6 、 件 5 7 的 垫 片 消 除 安 装 间 隙"
        toks, res, loss = predict(text, tag_list=tag_list)
        # toks, res, loss, metrics = predict(text, tag_list=tag_list,return_metrics=True)
        print("Tokens:", toks)
        print("Predicted:", res)
        print("True tags:", tag_list)
        print("Loss:", loss)
        # print("Metrics", metrics)

    elif mode == 'eval':
        evaluate_on_validation_set(val_path="./data/val.txt", method="batch")

    # 记录结束时间
    end_time = time.time()
    # 计算运行时间
    execution_time = end_time - start_time
    print(f"代码运行时间: {execution_time:.4f} 秒")
