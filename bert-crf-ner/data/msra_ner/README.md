---
license: Apache License 2.0
tags:
- NER
text:
  token-classification:    
    type:
      - ner
    language:
      - zh

---

# MSRA命名实体识别数据集
## 数据集概述
MSRA数据集是面向新闻领域的中文命名实体识别数据集。

### 数据集简介
本数据集包括训练集（46364）、测试集（4365），实体类型包括地名(LOC)、人名(NAME)、组织名(ORG)。

### 数据集的格式和结构
数据格式采用conll标准，数据分为两列，第一列是输入句中的词划分，第二列是每个词对应的命名实体类型标签。一个具体case的例子如下：

```
１	O
９	O
９	O
７	O
年	O
１	O
１	O
月	O
１	O
日	O

（	O
新	B-ORG
华	I-ORG
社	I-ORG
北	B-LOC
京	I-LOC
１	O
１	O
月	O
１	O
日	O
电	O
）	O

```

## 数据集版权信息

Creative Commons Attribution 4.0 International。

## 引用方式
```
@inproceedings{levow-2006-third,
    title = "The Third International {C}hinese Language Processing Bakeoff: Word Segmentation and Named Entity Recognition",
    author = "Levow, Gina-Anne",
    booktitle = "Proceedings of the Fifth {SIGHAN} Workshop on {C}hinese Language Processing",
    month = jul,
    year = "2006",
    address = "Sydney, Australia",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/W06-0115",
    pages = "108--117",
}
```