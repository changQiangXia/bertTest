"""
BERT+CRF模型定义

模型结构：
1. BERT编码器：提取上下文特征，输出每个token的隐藏状态
2. Dropout：防止过拟合
3. 线性层：将BERT输出映射到标签空间（发射分数）
4. CRF层：建模标签依赖，解码最优序列

为什么使用CRF而不是简单的Softmax？
- Softmax独立预测每个位置的标签，忽略了标签间的约束关系
- CRF通过转移矩阵学习标签间的依赖（如B-PER后面只能是I-PER或O）
- CRF使用Viterbi算法进行全局最优解码
"""
import torch
import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, BertConfig
from typing import Optional, Tuple, List


class BertCRFForNER(BertPreTrainedModel):
    """
    BERT+CRF命名实体识别模型
    
    继承自BertPreTrainedModel以便使用Hugging Face的预训练权重加载机制
    """
    
    def __init__(self, config: BertConfig, num_labels: int):
        """
        初始化模型
        
        Args:
            config: BERT配置
            num_labels: 标签数量（BIO格式下的标签类别数）
        """
        super().__init__(config)
        self.num_labels = num_labels
        
        # BERT编码器
        self.bert = BertModel(config, add_pooling_layer=False)
        
        # Dropout层
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # 分类器：将BERT输出映射到标签空间
        # 输出形状: (batch_size, seq_len, num_labels)
        self.classifier = nn.Linear(config.hidden_size, num_labels)
        
        # CRF层（手动实现，不通过nn.Module添加以避免参数重复）
        # 转移矩阵: transitions[i, j] 表示从标签i到标签j的转移分数
        self.transitions = nn.Parameter(torch.randn(num_labels, num_labels))
        
        # 初始化转移矩阵
        self._init_transitions()
        
        # 初始化权重
        self.post_init()
    
    def _init_transitions(self):
        """初始化CRF转移矩阵，设置非法转移为极小值"""
        nn.init.normal_(self.transitions, mean=0, std=0.1)
        
        # 注意：这里的约束根据具体标签集调整
        # 假设标签顺序: O, B-PER, I-PER, B-LOC, I-LOC, B-ORG, I-ORG
        # 0=O, 1=B-PER, 2=I-PER, 3=B-LOC, 4=I-LOC, 5=B-ORG, 6=I-ORG
        
        with torch.no_grad():
            # 不允许: I-XX 直接到 B-YY（不同类型实体之间必须有O分隔）
            # I-PER(2) 不能到 B-LOC(3), B-ORG(5)
            self.transitions[2, 3] = -1000
            self.transitions[2, 5] = -1000
            
            # I-LOC(4) 不能到 B-PER(1), B-ORG(5)
            self.transitions[4, 1] = -1000
            self.transitions[4, 5] = -1000
            
            # I-ORG(6) 不能到 B-PER(1), B-LOC(3)
            self.transitions[6, 1] = -1000
            self.transitions[6, 3] = -1000
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, ...]:
        """
        前向传播
        
        Args:
            input_ids: 输入token IDs，形状 (batch_size, seq_len)
            attention_mask: 注意力掩码，形状 (batch_size, seq_len)
            token_type_ids: token类型ID，形状 (batch_size, seq_len)
            labels: 真实标签，形状 (batch_size, seq_len)
            return_dict: 是否返回字典格式
        
        Returns:
            如果labels不为None，返回(loss, logits)
            否则返回(logits,)
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # BERT编码
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=return_dict,
        )
        
        # 获取最后一层隐藏状态
        sequence_output = outputs[0]  # (batch_size, seq_len, hidden_size)
        
        # Dropout
        sequence_output = self.dropout(sequence_output)
        
        # 发射分数（emission scores）
        emissions = self.classifier(sequence_output)  # (batch_size, seq_len, num_labels)
        
        # 如果提供了标签，计算损失
        if labels is not None:
            # 计算CRF损失
            loss = self._crf_loss(emissions, labels, attention_mask)
            return (loss, emissions)
        
        return (emissions,)
    
    def _crf_loss(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        计算CRF负对数似然损失
        
        损失 = log(sum(exp(所有路径分数))) - 真实路径分数
        """
        if mask is None:
            mask = torch.ones_like(tags, dtype=torch.bool)
        
        # 处理-100标签（忽略特殊token如[CLS], [SEP], [PAD]）
        # 将-100替换为0，并在mask中标记为无效
        valid_tags = tags.clone()
        valid_tags[valid_tags == -100] = 0
        
        # 计算真实路径分数（分子）
        gold_score = self._score_sentence(emissions, valid_tags, mask)
        
        # 计算配分函数（分母）
        forward_score = self._forward_algorithm(emissions, mask)
        
        # 负对数似然
        loss = forward_score - gold_score
        return loss.mean()
    
    def _score_sentence(
        self,
        emissions: torch.Tensor,
        tags: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        计算给定标签序列的分数
        
        序列分数 = 发射分数之和 + 转移分数之和
        """
        batch_size, seq_len = tags.shape
        
        # 发射分数
        emit_scores = emissions.gather(2, tags.unsqueeze(-1)).squeeze(-1)
        emit_scores = (emit_scores * mask).sum(dim=1)
        
        # 转移分数（相邻标签之间）
        trans_scores = 0
        for i in range(seq_len - 1):
            curr_tag = tags[:, i]
            next_tag = tags[:, i + 1]
            # 只有当两个位置都有效时才计算转移
            valid = mask[:, i] & mask[:, i + 1]
            trans_scores += self.transitions[curr_tag, next_tag] * valid
        
        return emit_scores + trans_scores
    
    def _forward_algorithm(
        self,
        emissions: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        前向算法计算配分函数（所有可能路径的分数之和）
        
        使用动态规划高效计算，避免枚举所有指数级路径
        """
        batch_size, seq_len, num_tags = emissions.shape
        
        # 初始化：第一个位置的发射分数
        alpha = emissions[:, 0].clone()  # (batch_size, num_tags)
        
        # 迭代计算后续位置
        for i in range(1, seq_len):
            # 扩展维度以便广播
            # alpha: (batch_size, num_tags) -> (batch_size, num_tags, 1)
            alpha_expanded = alpha.unsqueeze(2)
            
            # 转移矩阵: (num_tags, num_tags) -> (1, num_tags, num_tags)
            trans_expanded = self.transitions.unsqueeze(0)
            
            # 发射分数: (batch_size, num_tags) -> (batch_size, 1, num_tags)
            emit_expanded = emissions[:, i].unsqueeze(1)
            
            # 计算所有可能转移的分数: (batch_size, num_tags, num_tags)
            scores = alpha_expanded + trans_expanded + emit_expanded
            
            # 对所有前一状态求和（log-sum-exp技巧用于数值稳定）
            new_alpha = torch.logsumexp(scores, dim=1)
            
            # 只更新有效位置
            mask_i = mask[:, i].unsqueeze(-1).bool()
            alpha = torch.where(mask_i, new_alpha, alpha)
        
        # 对最后一个位置的所有标签求和
        return torch.logsumexp(alpha, dim=1)
    
    def decode(self, emissions: torch.Tensor, mask: torch.Tensor) -> List[List[int]]:
        """
        使用Viterbi算法解码最优标签序列
        
        Args:
            emissions: 发射分数 (batch_size, seq_len, num_labels)
            mask: 有效位置掩码 (batch_size, seq_len)
        
        Returns:
            最优标签序列列表
        """
        batch_size, seq_len, num_tags = emissions.shape
        
        # Viterbi变量：记录到每个位置每个标签的最大分数
        viterbi = emissions[:, 0].clone()  # (batch_size, num_tags)
        
        # 回溯指针：记录最优路径
        backpointers = []
        
        # 迭代计算
        for i in range(1, seq_len):
            # 扩展viterbi分数: (batch_size, num_tags) -> (batch_size, num_tags, 1)
            viterbi_exp = viterbi.unsqueeze(2)
            
            # 转移矩阵: (num_tags, num_tags) -> (1, num_tags, num_tags)
            trans = self.transitions.unsqueeze(0)
            
            # 计算所有转移的分数
            scores = viterbi_exp + trans  # (batch_size, num_tags, num_tags)
            
            # 找出最佳前一个标签
            best_scores, best_tags = scores.max(dim=1)  # (batch_size, num_tags)
            
            # 加上发射分数
            viterbi_new = best_scores + emissions[:, i]
            
            # 只更新有效位置
            mask_i = mask[:, i].unsqueeze(-1)
            viterbi = torch.where(mask_i, viterbi_new, viterbi)
            
            # 保存回溯指针
            backpointers.append(best_tags)
        
        # 找到最佳结束标签
        best_scores, best_last_tags = viterbi.max(dim=1)
        
        # 回溯构建最优路径
        best_paths = []
        for b in range(batch_size):
            path = [best_last_tags[b].item()]
            
            # 回溯
            for backptr in reversed(backpointers):
                prev_tag = backptr[b, path[0]].item()
                path.insert(0, prev_tag)
            
            best_paths.append(path)
        
        return best_paths
    
    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> List[List[int]]:
        """
        预测标签序列（推理时使用）
        
        Args:
            input_ids: 输入token IDs
            attention_mask: 注意力掩码
        
        Returns:
            预测的标签序列列表
        """
        self.eval()
        with torch.no_grad():
            # 获取发射分数
            outputs = self.forward(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            emissions = outputs[0]
            
            if attention_mask is None:
                attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
            
            # Viterbi解码
            predictions = self.decode(emissions, attention_mask)
        
        return predictions
