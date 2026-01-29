"""
CRF（条件随机场）层实现

CRF层的作用：
1. 建模标签间的依赖关系（如B-PER后只能接I-PER或O）
2. 使用Viterbi算法解码全局最优标签序列
3. 计算条件概率用于训练

数学原理：
- 转移矩阵（transitions）：存储标签i转移到标签j的分数
- 发射分数（emissions）：BERT输出的每个token对应各标签的分数
- 序列分数 = 发射分数之和 + 转移分数之和
"""
import torch
import torch.nn as nn
from typing import List, Optional, Tuple


class CRF(nn.Module):
    """
    线性链条件随机场（Linear Chain CRF）实现
    
    用于序列标注任务，建模标签间的转移约束
    """
    
    def __init__(self, num_tags: int, batch_first: bool = True):
        """
        初始化CRF层
        
        Args:
            num_tags: 标签数量（包括START和END）
            batch_first: 输入是否为batch_first格式（默认True）
        """
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # 转移矩阵：transitions[i, j] 表示从标签i转移到标签j的分数
        # 形状: (num_tags, num_tags)
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # START和END标签的索引
        self.START_TAG = num_tags
        self.END_TAG = num_tags + 1
        
        # 重新初始化转移矩阵，包含START和END
        self.transitions = nn.Parameter(torch.randn(num_tags + 2, num_tags + 2))
        
        # 约束：某些转移是不允许的，设为极小值
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化转移矩阵参数"""
        nn.init.normal_(self.transitions, mean=0, std=0.1)
        
        # START不能作为任何标签的下一个状态（除了真实的第一个标签）
        self.transitions.data[self.START_TAG, :] = -10000
        # 任何标签不能以END作为前一个状态（除了真实的最后一个标签）
        self.transitions.data[:, self.END_TAG] = -10000
        # START不能直接到END
        self.transitions.data[self.START_TAG, self.END_TAG] = -10000
    
    def forward(
        self, 
        emissions: torch.Tensor, 
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """
        计算负对数似然损失（用于训练）
        
        Args:
            emissions: BERT输出的发射分数，形状 (batch_size, seq_len, num_tags)
            tags: 真实标签，形状 (batch_size, seq_len)
            mask: 有效位置掩码，形状 (batch_size, seq_len)
            reduction: 损失缩减方式，"mean", "sum", 或 "none"
        
        Returns:
            负对数似然损失
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        # 计算真实路径的分数（gold score）
        numerator = self._compute_score(emissions, tags, mask)
        
        # 计算所有可能路径的分母（partition function）
        denominator = self._compute_normalizer(emissions, mask)
        
        # 负对数似然 = 分母 - 分子 = log(sum(exp(所有路径))) - 真实路径分数
        llh = denominator - numerator
        
        if reduction == "mean":
            return llh.mean()
        elif reduction == "sum":
            return llh.sum()
        else:
            return llh
    
    def _compute_score(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算给定标签序列的分数（分子）
        
        分数 = 所有位置的发射分数之和 + 所有位置的转移分数之和
        """
        batch_size, seq_len = tags.shape
        
        # 发射分数：从emissions中取出对应位置的分数
        # emissions[i, j, tags[i, j]] 表示第i个样本第j个位置的真实标签分数
        emissions_score = (emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1) * mask).sum(dim=1)
        
        # 转移分数
        # 首先添加START标签
        start_tags = torch.full((batch_size, 1), self.START_TAG, dtype=torch.long, device=tags.device)
        tags_with_start = torch.cat([start_tags, tags], dim=1)
        
        # 计算相邻标签间的转移分数
        transitions_score = 0
        for i in range(seq_len):
            # 只有当当前位置有效时才计算转移
            if i == 0:
                # 从START到第一个标签
                curr_trans = self.transitions[self.START_TAG, tags[:, i]]
            else:
                # 从前一个标签到当前标签
                curr_trans = self.transitions[tags_with_start[:, i], tags[:, i]]
            transitions_score += curr_trans * mask[:, i]
        
        # 添加END标签的转移分数
        # 找到每个序列的最后一个有效位置
        seq_lengths = mask.sum(dim=1).long()
        end_tags = tags.gather(1, (seq_lengths - 1).unsqueeze(-1)).squeeze(-1)
        end_transitions = self.transitions[end_tags, self.END_TAG]
        
        return emissions_score + transitions_score + end_transitions
    
    def _compute_normalizer(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        使用前向算法计算配分函数（分母）
        
        使用动态规划高效计算所有可能路径的分数之和
        """
        batch_size, seq_len, num_tags = emissions.shape
        
        # 初始化：第一个位置的分数（从START转移过来）
        # alpha[i, j] 表示第i个样本第j个标签的前向分数
        alpha = self.transitions[self.START_TAG, :num_tags].unsqueeze(0) + emissions[:, 0]
        
        # 迭代计算后续位置
        for i in range(1, seq_len):
            # 扩展alpha以便广播计算
            # alpha: (batch_size, num_tags) -> (batch_size, num_tags, 1)
            alpha_expanded = alpha.unsqueeze(2)
            
            # 转移分数: (batch_size, num_tags, num_tags)
            trans_expanded = self.transitions[:num_tags, :num_tags].unsqueeze(0)
            
            # 发射分数: (batch_size, 1, num_tags)
            emit_expanded = emissions[:, i].unsqueeze(1)
            
            # 计算所有可能转移的分数
            # scores: (batch_size, num_tags, num_tags)
            scores = alpha_expanded + trans_expanded + emit_expanded
            
            # log-sum-exp技巧稳定计算
            new_alpha = torch.logsumexp(scores, dim=1)
            
            # 只更新有效位置
            mask_expanded = mask[:, i].unsqueeze(-1)
            alpha = torch.where(mask_expanded, new_alpha, alpha)
        
        # 添加转移到END的分数
        alpha = alpha + self.transitions[:num_tags, self.END_TAG].unsqueeze(0)
        
        # 对所有可能的结束标签求和
        return torch.logsumexp(alpha, dim=1)
    
    def decode(self, emissions: torch.Tensor, mask: Optional[torch.Tensor] = None) -> List[List[int]]:
        """
        使用Viterbi算法解码最优标签序列
        
        Args:
            emissions: 发射分数，形状 (batch_size, seq_len, num_tags)
            mask: 有效位置掩码
        
        Returns:
            最优标签序列列表，每个元素是一个样本的标签序列
        """
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            if mask is not None:
                mask = mask.transpose(0, 1)
        
        if mask is None:
            mask = torch.ones(emissions.shape[:2], dtype=torch.bool, device=emissions.device)
        
        batch_size, seq_len, num_tags = emissions.shape
        
        # 初始化Viterbi变量
        # viterbi[i, j] 表示到位置i标签j的最大分数
        viterbi = self.transitions[self.START_TAG, :num_tags].unsqueeze(0) + emissions[:, 0]
        
        # 记录路径（回溯用）
        backpointers = []
        
        # 迭代计算
        for i in range(1, seq_len):
            # 扩展viterbi分数
            viterbi_expanded = viterbi.unsqueeze(2)  # (batch_size, num_tags, 1)
            trans_expanded = self.transitions[:num_tags, :num_tags].unsqueeze(0)  # (1, num_tags, num_tags)
            
            # 计算所有转移的分数
            scores = viterbi_expanded + trans_expanded  # (batch_size, num_tags, num_tags)
            
            # 找出最佳前一个标签
            best_scores, best_tags = scores.max(dim=1)  # (batch_size, num_tags)
            
            # 加上发射分数
            viterbi_new = best_scores + emissions[:, i]
            
            # 只更新有效位置
            mask_expanded = mask[:, i].unsqueeze(-1)
            viterbi = torch.where(mask_expanded, viterbi_new, viterbi)
            
            # 保存回溯指针
            backpointers.append(best_tags)
        
        # 添加END转移
        viterbi = viterbi + self.transitions[:num_tags, self.END_TAG].unsqueeze(0)
        
        # 找出最佳结束标签
        best_scores, best_last_tags = viterbi.max(dim=1)
        
        # 回溯得到最优路径
        best_paths = []
        for b in range(batch_size):
            path = [best_last_tags[b].item()]
            seq_len_b = mask[b].sum().item()
            
            # 回溯
            for i in range(seq_len_b - 1, 0, -1):
                best_tag = backpointers[i - 1][b, path[0]].item()
                path.insert(0, best_tag)
            
            best_paths.append(path)
        
        return best_paths


# 简化的CRF实现（用于实际训练，更高效）
class CRFLayer(nn.Module):
    """
    简化版CRF层，使用更高效的实现
    """
    
    def __init__(self, num_tags: int, batch_first: bool = True):
        super().__init__()
        self.num_tags = num_tags
        self.batch_first = batch_first
        
        # 转移矩阵
        self.transitions = nn.Parameter(torch.randn(num_tags, num_tags))
        
        # 初始化
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.normal_(self.transitions, mean=0, std=0.1)
    
    def forward(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor,
        reduction: str = "mean"
    ) -> torch.Tensor:
        """计算损失"""
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            tags = tags.transpose(0, 1)
            mask = mask.transpose(0, 1)
        
        # 计算真实路径分数
        gold_score = self._score_sentence(emissions, tags, mask)
        
        # 计算所有路径的分数
        forward_score = self._forward_alg(emissions, mask)
        
        # 负对数似然
        loss = forward_score - gold_score
        
        if reduction == "mean":
            return loss.mean()
        elif reduction == "sum":
            return loss.sum()
        return loss
    
    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """计算给定标签序列的分数"""
        batch_size, seq_len = tags.shape
        
        # 发射分数
        emit_scores = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        emit_scores = (emit_scores * mask).sum(dim=1)
        
        # 转移分数
        trans_scores = 0
        for i in range(seq_len - 1):
            curr_tag = tags[:, i]
            next_tag = tags[:, i + 1]
            trans_scores += self.transitions[curr_tag, next_tag] * mask[:, i + 1]
        
        return emit_scores + trans_scores
    
    def _forward_alg(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """前向算法计算配分函数"""
        batch_size, seq_len, num_tags = emissions.shape
        
        # 初始化
        alpha = emissions[:, 0].clone()
        
        for i in range(1, seq_len):
            emit = emissions[:, i].unsqueeze(1)  # (batch, 1, num_tags)
            trans = self.transitions.unsqueeze(0)  # (1, num_tags, num_tags)
            alpha_expand = alpha.unsqueeze(2)  # (batch, num_tags, 1)
            
            scores = alpha_expand + trans + emit  # (batch, num_tags, num_tags)
            new_alpha = torch.logsumexp(scores, dim=1)
            
            # 只更新有效位置
            mask_i = mask[:, i].unsqueeze(-1)
            alpha = torch.where(mask_i, new_alpha, alpha)
        
        return torch.logsumexp(alpha, dim=1)
    
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Viterbi解码"""
        if not self.batch_first:
            emissions = emissions.transpose(0, 1)
            mask = mask.transpose(0, 1)
        
        batch_size, seq_len, num_tags = emissions.shape
        
        # Viterbi变量
        viterbi = emissions[:, 0].clone()
        backpointers = []
        
        for i in range(1, seq_len):
            viterbi_exp = viterbi.unsqueeze(2)
            trans = self.transitions.unsqueeze(0)
            scores = viterbi_exp + trans  # (batch, num_tags, num_tags)
            
            best_scores, best_tags = scores.max(dim=1)
            viterbi_new = best_scores + emissions[:, i]
            
            mask_i = mask[:, i].unsqueeze(-1)
            viterbi = torch.where(mask_i, viterbi_new, viterbi)
            
            backpointers.append(best_tags)
        
        # 找到最佳结束标签
        best_scores, best_tags = viterbi.max(dim=1)
        
        # 回溯构建路径
        best_paths = [best_tags.unsqueeze(1)]
        for backptr in reversed(backpointers):
            best_tags = backptr.gather(1, best_tags.unsqueeze(-1)).squeeze(-1)
            best_paths.insert(0, best_tags.unsqueeze(1))
        
        return torch.cat(best_paths, dim=1)
