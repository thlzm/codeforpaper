# Incontext learning
上下文小样本学习范式，通过给模型输入任务提示和相关示例使用预训练语言模型，无需重新训练模型和增加额外参数，更接近人类解决新任务的机制。

本目录为鹏程预训练大模型的Incontext learning代码仓库，可以用于模型Incontext learning的多机多卡测试。

## 实例
### 实例1: 鹏程·盘古 句子关系-NLI任务
目前已支持鹏程·盘古模型的在NLI任务的Incontext learning，任务实例为CMNLI(自然语言推理中文版)任务。

- **算法原理**
    * 将输入嵌入到prompt中同时加上示例输入给模型，类似人类解决新任务的机制。
    * 分类类型任务：将任务分解为困惑度比较任务。
    * 具体请参参考盘古技术报告："PANGU-α: LARGE-SCALE AUTOREGRESSIVE PRETRAINED CHINESE LANGUAGE MODELS WITH AUTO-PARALLEL COMPUTATION"。
    
- **环境**
    * python依赖包见首页requirements.txt。
    * 推荐使用英伟达的官方 docker 镜像docker pull nvcr.io/nvidia/pytorch:20.03-py3，并安装NLTK。
    
- **模型**
    * 目前实现基于盘古2.6\13B的pytorch-GPU版。
    * 模型下载，请参考[PanGu-Alpha-GPU](https:\//git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha-GPU)。
    
- **运行**
    * 参考脚本：scripts/incontext_ppl_nli.sh
    
- **性能**
    * 盘古2.6B-fp16，zero shot，验证集acc: 45%左右。