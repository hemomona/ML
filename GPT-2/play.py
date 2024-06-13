#!/usr/bin/env python
# -*- coding:utf-8 -*-

"""
# File       : play.py
# Time       ：2024/6/12 16:34
# Author     ：Jago
# Email      ：huwl2022@shanghaitech.edu.cn
# version    ：python 3.10.11
# Description：
https://blog.csdn.net/yyh2508298730/article/details/138715878 轻松学会HuggingFace模型下载与保存
https://blog.csdn.net/weixin_43431218/article/details/135403324 【Hugggingface.co】关于huggingface.co无法访问
"""
import matplotlib.pyplot as plt
import transformers

from transformers import GPT2LMHeadModel  # transformers==4.42.0.dev0 huggingface_hub-0.23.3

model_hf = GPT2LMHeadModel.from_pretrained('dataroot/models/openai-community/gpt2')  # 124M parameters 523MB
sd_hf = model_hf.state_dict()  # raw tensors

# for k, v in sd_hf.items():
#     print(k, v.shape)
#
# transformer.wte.weight torch.Size([50257, 768]) weighted token embedding
# transformer.wpe.weight torch.Size([1024, 768]) weighted position embedding
# ...

print(sd_hf["transformer.wpe.weight"].view(-1)[:20])

plt.imshow(sd_hf["transformer.wpe.weight"], cmap="gray")
plt.show()
