# PanGu-Alpha-Applications

[中文](README.md)|English

## Introduction
This project aims to provide algorithm examples and application demonstrations from algorithm layer to application layer for Pengcheng series super large-scale pre training models, take the large model as AI infrastructure, and accelerate the application technology innovation and application ecology construction of the large model.

## Navigation
|  | Project | Describe |
| --- | :---------------- | :--------------- |
| pretrained models | [[PanGu-Alpha](https:\//git.openi.org.cn/PCL-Platform.Intelligence/PanGu-Alpha)]  <br> [[PanGu-Alpha-Evolution](./model/pangu_evolution)]| Pengcheng series pre-training models |
| algorithm&task | [[model transfer](./method)] <br> [[model compression](./method)] |The path of algorithm&task|
| application | [[application](./app)]        | The path of application |

## Progress
- **2021.08.18** <br>
  * Relase the first version of baseline for few shot model transfer.
  * Task layer：cmnli baseline.
  * Algorithm layer：fine-tune，prompt-tune，incontext-learning.

## Framework
**Model layer**：Training models based on distributed training, big data and efficient algorithm. Building AI underlying infrastructure in the fields of text, multilingual, multimodal and knowledge graph.<br>
**Algorithm layer**：Based on the pre-trained models, make algorithm innovation in model compression, small sample model migration and continuous learning, and build a basic algorithm module to provide underlying algorithm support for the landing application of the pre training model.<br>
**Task layer**：Based on the basic algorithm module of the algorithm layer, the implementation examples of the two basic tasks NLU and NLG are constructed to provide the underlying task modeling support for the upper application.<br>
**Application layer**：Applications such as dialogue robot, writing assistant, multilingual translation, Knowledge Q &amp; A and patent aided generation are designed to provide demonstration applications for the landing application of the pre training model and promote the accelerated landing and in-depth application of the model.

## Codes
- **structure**
```bash
PanGu-Alpha-Application
|-- README.md
|-- app    
|-- com    
|-- megatron   
|-- method     
|-- requirements.txt
`-- resource
```


## License
```bash
adding
```