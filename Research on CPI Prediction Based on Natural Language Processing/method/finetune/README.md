# Finetune
预训练微调范式，通过引入较多额外的参数并使用特定任务的目标函数对预训练模型进行微调，将预训练语言模型适应于不同的下游任务。

本目录为鹏程预训练大模型的fine-tune代码仓库，可以用于模型fine-tune的多机多卡训练/测试。

## 实例
### 实例1: 鹏程·盘古 句子关系-NLI任务
目前已支持鹏程·盘古模型的在NLI任务的fine-tune，具体实现实例为CMNLI(自然语言推理中文版)任务。

- **算法原理**
    * 因为鹏程·盘古模型不像BERT模型有CLS位置，所以没法获得句子的表示。
    * 处理方法是是将两个句子拼接起来，然后增加特殊token放句子末尾，通过finetune请特殊token获得整体表示，再接一个分类层，将NLI任务转换成三分类。

- **环境**
    * python依赖包见首页requirements.txt。
    * 推荐使用英伟达的官方 docker 镜像docker pull nvcr.io/nvidia/pytorch:20.03-py3，并安装NLTK。
    
- **模型**
    * 目前实现基于盘古2.6\13B的pytorch-GPU版。
    * 模型下载，请参考[PanGu-Alpha-GPU](https:\//git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha-GPU)。
    
- **运行**
    * 参考脚本：scripts/finetune_nli.sh
    
- **性能**
    * 盘古2.6B-fp16，未充分调优情况下，训练集覆盖6遍时，验证集acc: 50%左右。
    * 如充分调优，性能应该有较大提升空间。
