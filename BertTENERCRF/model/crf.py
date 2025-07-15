# coding=utf-8

import torch
import torch.nn as nn


def log_sum_exp(vec, dim=-1):
    """
    数值稳定的log sum exp计算
    Args:
        vec: 任意形状的张量
        dim: 要在哪个维度上求和，默认为最后一个维度
    """
    max_score, _ = vec.max(dim=dim, keepdim=True)
    return max_score.squeeze(dim) + torch.log(torch.sum(torch.exp(vec - max_score), dim=dim))


class CRF(nn.Module):
    def __init__(self, target_size, average_batch=True, use_cuda=True):
        super(CRF, self).__init__()

        self.target_size = target_size
        self.average_batch = average_batch
        self.use_cuda = use_cuda
        self.device = torch.device("cuda" if use_cuda else "cpu")

        # 转移矩阵 (target_size, target_size)
        # transitions[i][j] 表示从标签i转移到标签j的分数
        self.transitions = nn.Parameter(torch.zeros(target_size, target_size, device=self.device))

        # 定义特殊标签的索引
        self.START_TAG_IDX = 1  # <start>的索引
        self.END_TAG_IDX = target_size - 1  # <eos>的索引
        self.PAD_TAG_IDX = 0  # <pad>的索引

        # 初始化参数
        self._initialize_parameters()

    def _initialize_parameters(self):
        """初始化参数，包括BIOES约束和特殊标签处理"""
        # 使用较小的初始值，避免数值问题
        nn.init.uniform_(self.transitions, -0.1, 0.1)

        with torch.no_grad():
            # 设置特殊标签的转移约束
            # 不能转移到START
            self.transitions[:, self.START_TAG_IDX] = -10000.0
            # 不能从END转移
            self.transitions[self.END_TAG_IDX, :] = -10000.0
            # PAD的特殊处理
            self.transitions[self.PAD_TAG_IDX, :] = -10000.0
            self.transitions[:, self.PAD_TAG_IDX] = -10000.0
            # PAD可以转移到PAD
            self.transitions[self.PAD_TAG_IDX, self.PAD_TAG_IDX] = 0.0

            # 添加BIOES约束
            self._add_bioes_constraints()

    def _add_bioes_constraints(self):
        """添加BIOES标注规则约束到转移矩阵"""
        with torch.no_grad():
            # 使用适中的约束值
            IMPOSSIBLE = -1000.0

            # 直接硬编码标签列表和索引
            tag_list = [
                '<pad>', '<start>', 'O',
                'B-Pd', 'I-Pd', 'E-Pd', 'S-Pd',
                'B-Fm', 'I-Fm', 'E-Fm', 'S-Fm',
                'B-Co', 'I-Co', 'E-Co', 'S-Co',
                'B-Fe', 'I-Fe', 'E-Fe', 'S-Fe',
                'B-Df', 'I-Df', 'E-Df', 'S-Df',
                'B-Pf', 'I-Pf', 'E-Pf', 'S-Pf',
                'B-Pr', 'I-Pr', 'E-Pr', 'S-Pr',
                'B-Td1', 'I-Td1', 'E-Td1', 'S-Td1',
                'B-Ed', 'I-Ed', 'E-Ed', 'S-Ed',
                'B-Wd', 'I-Wd', 'E-Wd', 'S-Wd',
                'B-Dd', 'I-Dd', 'E-Dd', 'S-Dd',
                'B-Td2', 'I-Td2', 'E-Td2', 'S-Td2',
                'B-Wo', 'I-Wo', 'E-Wo', 'S-Wo',
                'B-Ws', 'I-Ws', 'E-Ws', 'S-Ws',
                'B-Ma', 'I-Ma', 'E-Ma', 'S-Ma',
                '<eos>'
            ]

            # 设置BIOES约束规则
            for i, from_tag in enumerate(tag_list):
                for j, to_tag in enumerate(tag_list):
                    # 确保不超出矩阵边界
                    if i >= self.target_size or j >= self.target_size:
                        continue

                    # 提取实体类型
                    from_prefix = from_tag.split('-')[0] if '-' in from_tag else from_tag
                    to_prefix = to_tag.split('-')[0] if '-' in to_tag else to_tag

                    from_type = from_tag.split('-')[1] if '-' in from_tag else ""
                    to_type = to_tag.split('-')[1] if '-' in to_tag else ""

                    # 应用BIOES约束规则

                    # 1. B-X只能接I-X或E-X(相同实体类型)
                    if from_prefix == 'B':
                        if to_prefix in ['I', 'E']:
                            if from_type != to_type:
                                self.transitions.data[i, j] = IMPOSSIBLE
                        elif to_prefix not in ['O', 'B', 'S', '<eos>']:
                            self.transitions.data[i, j] = IMPOSSIBLE

                    # 2. I-X只能接I-X或E-X(相同实体类型)
                    elif from_prefix == 'I':
                        if to_prefix in ['I', 'E']:
                            if from_type != to_type:
                                self.transitions.data[i, j] = IMPOSSIBLE
                        elif to_prefix not in ['O', 'B', 'S', '<eos>']:
                            self.transitions.data[i, j] = IMPOSSIBLE

                    # 3. E-X后面不能接I-X或E-X(任何实体类型)
                    elif from_prefix == 'E':
                        if to_prefix in ['I', 'E']:
                            self.transitions.data[i, j] = IMPOSSIBLE

                    # 4. S-X后面不能接I-X或E-X(任何实体类型)
                    elif from_prefix == 'S':
                        if to_prefix in ['I', 'E']:
                            self.transitions.data[i, j] = IMPOSSIBLE

                    # 5. <start>后面不能接I-X或E-X
                    elif from_tag == '<start>':
                        if to_prefix in ['I', 'E']:
                            self.transitions.data[i, j] = IMPOSSIBLE

                    # 6. O后面不能接I-X或E-X
                    elif from_tag == 'O':
                        if to_prefix in ['I', 'E']:
                            self.transitions.data[i, j] = IMPOSSIBLE

                    # 7. <eos>前面不能是B-X或I-X
                    if to_tag == '<eos>':
                        if from_prefix in ['B', 'I']:
                            self.transitions.data[i, j] = IMPOSSIBLE

    def _forward_alg(self, feats, mask):
        """
        前向算法计算配分函数Z(x)
        Args:
            feats: (batch_size, seq_len, num_tags) 发射分数
            mask: (batch_size, seq_len) 掩码，1表示真实token，0表示填充
        """
        batch_size, seq_len, num_tags = feats.size()

        # 初始化：考虑从START标签到各个标签的转移
        # 加上第一个位置的发射分数
        init_alphas = torch.full((batch_size, num_tags), -10000.0, device=self.device)
        init_alphas[:, self.START_TAG_IDX] = 0

        # 前向变量，存储每个位置上每个标签的所有路径得分的log-sum-exp
        forward_var = init_alphas

        # 依次处理每个位置
        for i in range(seq_len):
            # 当前位置的发射分数
            emit_score = feats[:, i]  # (batch_size, num_tags)

            # 计算当前位置的alphas值
            # next_tag_var[i][j] = forward_var[i][k] + trans_score[k][j]
            alphas_t = []
            for next_tag in range(num_tags):
                # 广播机制: forward_var + transitions[:, next_tag]
                next_tag_var = forward_var + self.transitions[:, next_tag].unsqueeze(0)
                alphas_t.append(log_sum_exp(next_tag_var, dim=1))

            alphas_t = torch.stack(alphas_t, dim=1)  # (batch_size, num_tags)

            # 加上发射分数
            next_forward_var = alphas_t + emit_score

            # 使用mask更新forward_var
            mask_i = mask[:, i].unsqueeze(1)  # (batch_size, 1)
            forward_var = mask_i * next_forward_var + (1 - mask_i) * forward_var

        # 考虑转移到END标签
        terminal_var = forward_var + self.transitions[:, self.END_TAG_IDX].unsqueeze(0)

        # 计算最终的分区函数Z(x)
        alpha = log_sum_exp(terminal_var, dim=1)  # (batch_size,)

        return alpha

    def _score_sentence(self, feats, mask, tags):
        """
        计算给定标签序列的分数
        Args:
            feats: (batch_size, seq_len, num_tags) 发射分数
            mask: (batch_size, seq_len) 掩码
            tags: (batch_size, seq_len) 标签序列
        """
        batch_size, seq_len = tags.size()

        # 初始化分数
        score = torch.zeros(batch_size, device=self.device)

        # 准备起始标签
        start_tags = torch.full((batch_size, 1), self.START_TAG_IDX,
                                dtype=torch.long, device=self.device)

        # 拼接start_tags和tags
        tags = torch.cat([start_tags, tags], dim=1)  # (batch_size, seq_len+1)

        # 依次处理每个位置，计算发射分数和转移分数
        for i in range(seq_len):
            # 当前位置有效时才计算分数
            valid_mask = mask[:, i]

            # 当前位置的发射分数
            emit_score = torch.zeros(batch_size, device=self.device)
            valid_indices = valid_mask.nonzero().squeeze(-1)
            if valid_indices.size(0) > 0:
                emit_score[valid_indices] = feats[valid_indices, i, tags[valid_indices, i + 1]]

            # 当前位置的转移分数
            trans_score = torch.zeros(batch_size, device=self.device)
            if valid_indices.size(0) > 0:
                trans_score[valid_indices] = self.transitions[tags[valid_indices, i],
                tags[valid_indices, i + 1]]

            # 更新总分数
            score = score + emit_score + trans_score

        # 加上转移到END标签的分数
        # 需要找到每个序列的最后一个有效位置
        seq_ends = mask.sum(dim=1).long() - 1  # (batch_size,)

        # 获取每个序列最后一个标签
        last_tags = tags[torch.arange(batch_size, device=self.device), seq_ends + 1]

        # 加上到END的转移分数
        end_trans = self.transitions[last_tags, self.END_TAG_IDX]
        score = score + end_trans

        return score

    def _viterbi_decode(self, feats, mask):
        """
        维特比算法解码最佳路径
        Args:
            feats: (batch_size, seq_len, num_tags) 发射分数
            mask: (batch_size, seq_len) 掩码
        Returns:
            best_scores: (batch_size,) 最佳路径的分数
            best_paths: 最佳路径的标签序列，长度为batch_size的列表
        """
        batch_size, seq_len, num_tags = feats.size()

        # 初始化: 考虑从START到各个标签的转移
        viterbi_vars = torch.full((batch_size, num_tags), -10000.0, device=self.device)
        viterbi_vars[:, self.START_TAG_IDX] = 0

        # 存储回溯指针
        backpointers = []

        # 沿着序列进行动态规划
        for i in range(seq_len):
            # 在当前位置计算每个标签的最佳前导标签
            bptrs_t = []
            viterbivars_t = []

            for next_tag in range(num_tags):
                # viterbi_vars加上转移分数
                next_tag_var = viterbi_vars + self.transitions[:, next_tag].unsqueeze(0)
                # 找到最佳前导标签
                best_tag_id = torch.argmax(next_tag_var, dim=1)
                bptrs_t.append(best_tag_id)
                viterbivars_t.append(next_tag_var.gather(1, best_tag_id.unsqueeze(1)).squeeze(1))

            # 将这一步的回溯指针加入列表
            backpointers.append(torch.stack(bptrs_t, dim=1))

            # 加上发射分数，得到新的viterbi变量
            viterbi_vars_new = torch.stack(viterbivars_t, dim=1) + feats[:, i]

            # 使用mask更新viterbi_vars
            mask_i = mask[:, i].unsqueeze(1)
            viterbi_vars = mask_i * viterbi_vars_new + (1 - mask_i) * viterbi_vars

        # 转移到END标签
        terminal_var = viterbi_vars + self.transitions[:, self.END_TAG_IDX].unsqueeze(0)
        best_tag_id = torch.argmax(terminal_var, dim=1)
        best_path_scores = terminal_var.gather(1, best_tag_id.unsqueeze(1)).squeeze(1)

        # 回溯找到最佳路径
        best_paths = []
        for b in range(batch_size):
            best_path = [best_tag_id[b].item()]

            # 计算有效序列长度
            seq_len_b = int(mask[b].sum().item())

            # 回溯
            for bptrs_t in reversed(backpointers[:seq_len_b]):
                best_path.append(bptrs_t[b][best_path[-1]].item())

            # 反转路径（去除START标签）
            best_path.reverse()
            if best_path[0] == self.START_TAG_IDX:
                best_path = best_path[1:]

            best_paths.append(best_path)

        return best_path_scores, best_paths

    def neg_log_likelihood_loss(self, feats, mask, tags):
        """
        计算负对数似然损失
        Args:
            feats: (batch_size, seq_len, num_tags) 发射分数
            mask: (batch_size, seq_len) 掩码
            tags: (batch_size, seq_len) 标签序列
        """
        batch_size = feats.size(0)

        # 计算前向算法得到的配分函数Z(x)
        forward_score = self._forward_alg(feats, mask)

        # 计算正确路径的分数
        gold_score = self._score_sentence(feats, mask, tags)

        # 计算损失：-log(P(y|x)) = -log(exp(score(x,y))/Z(x)) = log(Z(x)) - score(x,y)
        loss = forward_score - gold_score

        # 进行批量平均（如果设置了average_batch）
        if self.average_batch:
            loss = loss.mean()
        else:
            loss = loss.sum()

        return loss

    def neg_log_likelihood(self, feats, mask, tags):
        """兼容旧接口"""
        return self.neg_log_likelihood_loss(feats, mask, tags)

    def forward(self, feats, mask):
        """
        前向传播，返回维特比解码的最佳路径
        Args:
            feats: (batch_size, seq_len, num_tags) 发射分数
            mask: (batch_size, seq_len) 掩码
        """
        return self._viterbi_decode(feats, mask)