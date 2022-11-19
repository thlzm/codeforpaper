# Prompttune
预训练加提示的范式，这种范式不是让预训练的语言模型适应下游任务，而是形式化下游任务使之在提示(prompt)的帮助更适合预训练模型的原始任务来解决问题。

本目录为鹏程预训练大模型的prompt-tune代码仓库，可以用于模型prompt-tune的多机多卡训练/测试。

## 实例
### 实例1: 鹏程·盘古 PET 句子关系-NLI任务
目前已支持鹏程·盘古模型的在NLI任务的prompt-tune，算法实例基于PET的思想，任务实例为CMNLI(自然语言推理中文版)任务。

- **算法原理**
    * 将输入嵌入到prompt中，label映射到特定的token或词上，从而将任务转换成生成任务。
    * PET可以看作是一种手工prompt+tuning的方式，具体可参考论文：“It’s Not Just Size That Matters:Small Language Models Are Also Few-Shot Learners”。
    
- **环境**
    * python依赖包见首页requirements.txt。
    * 推荐使用英伟达的官方 docker 镜像docker pull nvcr.io/nvidia/pytorch:20.03-py3，并安装NLTK。
    
- **模型**
    * 目前实现基于盘古2.6\13B的pytorch-GPU版。
    * 模型下载，请参考[PanGu-Alpha-GPU](https:\//git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha-GPU)。
    
- **运行**
    * 参考脚本：scripts/prompttune_nli.sh
    
- **性能**
    * 盘古2.6B-fp16，未充分调优情况下，训练集覆盖6遍时，验证集acc: 55%左右。
    * 如充分调优，性能应该有较大提升空间。