<p align="center" width="100%">
<img src="./image.png" alt="OpenAlpaca" style="width: 50%; min-width: 300px; display: block; margin: auto;">
</p>

# OpenAlpaca: A Fully Open-Source Instruction-Following Model Based On OpenLLaMA

![Data License](https://img.shields.io/badge/Data_License-CC%20BY--SA%203.0-red.svg)
![Code License](https://img.shields.io/badge/Code%20License-Apache_2.0-green.svg)
![Model Weight License](https://img.shields.io/badge/Model_Weights%20License-Apache_2.0-green.svg)
![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)

**Team:** [Yixuan Su](https://yxuansu.github.io/)<sup>\*</sup>, [Tian Lan](https://github.com/gmftbyGMFTBY)<sup>\*</sup>, and [Deng Cai](https://jcyk.github.io/) (The first two members<sup>\*</sup> contributed equally.)

This is the repo for the OpenAlpaca project, which aims to build and share an instruction-following model based on [OpenLLaMA](https://github.com/openlm-research/open_llama). We note that, following OpenLLaMA, OpenAlpaca is permissively licensed under [the Apache 2.0 license](https://www.apache.org/licenses/LICENSE-2.0). This repo contains

- The <a href='#data'>data</a> used for fine-tuning the model.
- The <a href='#code'>code</a> for fine-tuning the model.
- The <a href='#weights'>weights</a> for the fine-tuned model.
- The <a href='#example_usage'>example usage</a> of OpenAlpaca.

**Usage and License Notices:** OpenAlpaca follows the distribution permission of [OpenLLaMA](https://github.com/openlm-research/open_llama), i.e. the Apache 2.0 license, which means OpenAlpaca can be used in any academic or commercial purposes for free.


****

<span id='data'/>

# Data:

The data, i.e. [openalpaca.json](https://github.com/yxuansu/OpenAlpaca/blob/main/openalpaca.json), we use to fine-tune the model contains ~15k instances and is constructed from the [databricks-dolly-15k dataset](https://huggingface.co/datasets/databricks/databricks-dolly-15k) by removing samples that are too long. Following the original databricks-dolly-15k dataset, our data is also licensed under [the CC BY-SA 3.0 license](https://repositories.lib.utexas.edu/bitstream/handle/2152/11616/license_text?sequence=2&isAllowed=y) which allows it to be used in any academic and commerical purposes. 

**Format:** Following [Stanford Alpaca](https://github.com/tatsu-lab/stanford_alpaca), our json file is a list of dictionaries, each one contains the following fields.
- `instruction`: it describes the task the model should perform.
- `input`: optional context or input for the task (e.g. the document for summarization task). 
- `output`: the answer to the instruction (and the optional input) which is written by human.

**Reproduce the data:** To reproduce the data, simply run `python3 process_dataset.py`.

<span id='weights'/>

# Model Weights:

|**Model Name**|**Model Card**|**Model Description**|
|:-------------:|:-------------:|:-------------:|
|`openllmplayground/openalpaca_7b_preview_2bt`|[[Link]](https://huggingface.co/openllmplayground/openalpaca_7b_preview_2bt/)|```The OpenAlpaca model fine-tuned from the previewed version of OpenLLaMA that is trained with 200 billion tokens.```|



<span id='example_usage'/>

# Example Usage:

Below shows an example on how to use OpenAlpaca

```python
import torch
from transformers import LlamaForCausalLM, LlamaTokenizer

# the previewed version of OpenAlpaca
model_path = r'openllmplayground/openalpaca_7b_preview_2bt' 
tokenizer = LlamaTokenizer.from_pretrained(model_path)
model = LlamaForCausalLM.from_pretrained(model_path).cuda()

# same prompt as provided in https://crfm.stanford.edu/2023/03/13/alpaca.html
instruction = r'What is an alpaca? How is it different from a llama?'
'''
instruction = r'Write an e-mail to congratulate new Standford admits and mention that you are excited about meeting all of them in person.'
instruction = r'What is the capital of Tanzania?'
instruction = r'Write a well-thought out abstract for a machine learning paper that proves that 42 is the optimal seed for training neural networks.'
'''

prompt_no_input = f'### Instruction:\n{instruction}\n\n### Response:'
tokens = tokenizer.encode(prompt_no_input)
bos_token_id, eos_token_id = 1, 2 # see https://github.com/openlm-research/open_llama#preview-weights-release-and-usage
tokens = [bos_token_id] + tokens + [eos_token_id] + [bos_token_id]
tokens = torch.LongTensor(tokens[-1024:]).unsqueeze(0).cuda()
instance = {'input_ids': tokens,
            'top_k': 50,
            'top_p': 0.9,
            'generate_len': 128}
            
length = len(tokens[0])
with torch.no_grad():
    rest = model.generate(
            input_ids=tokens, 
            max_length=length+instance['generate_len'], 
            use_cache=True, 
            do_sample=True, 
            top_p=instance['top_p'], 
            top_k=instance['top_k']
        )

output = rest[0][length:]
string = tokenizer.decode(output, skip_special_tokens=False)
string = string.replace('<s>', '').replace('</s>', '').strip()
print(f'[!] Generation results: {string}')
```

**[Model Output]**
```
[!] Generation results: An alpaca is a smaller version of a llama. Both are South American animals. However, 
an alpaca has a wooly coat that is very soft, while a llama’s coat is tougher and woolier.
```


<span id='future_plans'/>

<span id='code'/>

# Fine-tuning the Model:

To be released soon.

# Future Plans:

1. The current `openalpaca_7b_preview_2bt` model is fine-tuned on the previewed version of [OpenLLaMA-7b](https://github.com/openlm-research/open_llama). The previewed version of OpenLLaMA-7b is only trained with **200** billion tokens and we expect the performance of the base OpenLLaMA-7b model to improve as the training continues. We will update the version of OpenAlpaca so long as newer checkpoint is released by the authors of OpenLLaMA.

2. We also plan to do a rigorous evaluation of OpenAlpaca and compare it with other publicly accessible models.

<span id='reference'/>

# Reference:

If you found OpenAlpaca useful in your research or applications, please kindly cite using the following BibTeX:
```
@misc{openalpaca,
  author = {Yixuan Su and Tian Lan and Deng Cai},
  title = {OpenAlpaca: A Fully Open-Source Instruction-Following Model Based On OpenLLaMA},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/yxuansu/OpenAlpaca}},
}
```
```
@software{openlm2023openllama,
  author = {Xinyang Geng and Hao Liu},
  title = {OpenLLaMA: An Open Reproduction of LLaMA},
  month = May,
  year = 2023,
  url = {https://github.com/openlm-research/open_llama}
}
```
```
@misc{alpaca,
  author = {Rohan Taori and Ishaan Gulrajani and Tianyi Zhang and Yann Dubois and Xuechen Li and Carlos Guestrin and Percy Liang and Tatsunori B. Hashimoto },
  title = {Stanford Alpaca: An Instruction-following LLaMA model},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/tatsu-lab/stanford_alpaca}},
}
```
```
@article{touvron2023llama,
  title={Llama: Open and efficient foundation language models},
  author={Hugo Touvron and Thibaut Lavril and Gautier Izacard and Xavier Martinet and Marie{-}Anne Lachaux and Timoth{\'{e}}e Lacroix and Baptiste Rozi{\`{e}}re and Naman Goyal and Eric Hambro and Faisal Azhar and Aur{\'{e}}lien Rodriguez and Armand Joulin and Edouard Grave and Guillaume Lample},
  journal={arXiv preprint arXiv:2302.13971},
  year={2023}
}
```

<span id='acknowledgements'/>

# Acknowledgements:

This repo benefits from [OpenLLaMA](https://github.com/openlm-research/open_llama), [Alpaca](https://github.com/tatsu-lab/stanford_alpaca), and [Databricks](https://www.databricks.com/). Thanks for their wonderful works!
