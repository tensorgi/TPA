# T6: Tensor ProducT ATTenTion Transformer

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.4.0-orange.svg)

T6 (**Tensor ProducT ATTenTion Transformer**) is a state-of-the-art transformer model that leverages Tensor Product Attention (TPA) mechanisms to enhance performance and reduce KV cache size. This repository provides tools for data preparation, model pretraining, and evaluation to facilitate research and development using the T6 architecture.

This repository contains the official code for the paper "[Tensor Product Attention Is All You Need](https://arxiv.org/abs/2501.06425)".

Authors: [Yifan Zhang](https://yifzhang.com)\*, [Yifeng Liu](https://lauyikfung.github.io)\*, [Huizhuo Yuan](https://scholar.google.com/citations?user=8foZzX4AAAAJ), [Zhen Qin](https://doraemonzzz.com), [Yang Yuan](https://scholar.google.com/citations?user=7o4wtKEAAAAJ&hl=en), [Quanquan Gu](https://web.cs.ucla.edu/~qgu/), [Andrew Chi-Chih Yao](https://en.wikipedia.org/wiki/Andrew_Yao)

[[Webpage](https://tensorgi.github.io/T6)] [[Huggingface](https://huggingface.co/papers/2501.06425)]

## Table of Contents

- [Features](#features)
- [Hardware Requirements](#hardware-Requirement)
- [Installation](#installation)
- [Data Preparation](#data-preparation)
  - [Fineweb-Edu-100B](#fineweb-edu-100b)
  - [OpenWebText](#openwebtext)
- [Pretraining](#pretraining)
- [Evaluation](#evaluation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)
- [Acknowledgements](#acknowledgements)

## Features

- **Tensor Product Attention:** Implements advanced attention mechanisms for improved model performance.
- **Scalability:** Efficient training procedures optimized for large-scale datasets and multi-GPU setups.
- **Flexible Data Support:** Compatible with popular datasets like [Fineweb-Edu-100B](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/) and [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/).
- **Comprehensive Evaluation:** Integrated with [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) for standardized benchmarking.
- **Higher-order TPA** (TBD): [Higher-order TPA](./Higher_order_TPA.pdf).
- **Flash TPA** (TBD): [Flash TPA](./Flash_TPA.pdf).

## Hardware Requirements
A100 and H100 are recommended. At least 8*80G VRAM is needed.

## Installation

Ensure you have Python 3.10 or higher installed. It's recommended to use a virtual environment to manage dependencies.

1. **Clone the Repository**

   ```bash
   git clone https://github.com/tensorgi/T6.git
   cd T6
   ```
2. **Create and Activate a Virtual Environment**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install Required Packages**

   ```bash
   pip install torch==2.4.0 numpy transformers datasets tiktoken wandb tqdm
   ```

## Data Preparation

Prepare the necessary datasets before pretraining the model. T6 supports both [Fineweb-Edu-100B](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/) and [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/).

### Fineweb-Edu-100B

Fineweb-Edu-100B is a large-scale educational dataset hosted on Hugging Face.

1. **Navigate to the Data Directory**

   ```bash
   cd data/fineweb-edu
   ```
2. **Run the Data Preparation Script**

   ```bash
   python fineweb-edu.py
   ```
3. **Move the Prepared Data**

   ```bash
   mv fineweb-edu100B ..
   cd ../..
   ```

### OpenWebText

OpenWebText is an open reproduction of OpenAI's WebText dataset.

1. **Run the Data Preparation Script**

   ```bash
   python data/openwebtext/prepare.py
   ```

   *Ensure you have sufficient storage and computational resources as OpenWebText is sizable.*

## Pretraining

Pretrain the T6 model using the prepared datasets. The provided scripts support distributed training across multiple GPUs.

1. **Using the Provided Bash Script**

   Execute the pretraining script which handles the training process.

   ```bash
   bash pretrain.sh
   ```
2. **Manual Execution with `torchrun`**

   For more control or customization, use `torchrun` to initiate training. Replace `config/train_T6_medium_adam_80g8.py` with your desired configuration file.

   ```bash
   torchrun --standalone --nproc_per_node=8 \
       train_adam_fw.py \
       config/train_T6_medium_adam_80g8.py
   ```

   - `--nproc_per_node=8` specifies the number of processes (typically matching the number of GPUs).

## Evaluation

Evaluate the performance of the pretrained T6 model using standardized benchmarks.

1. **Navigate to the Evaluation Harness Directory**

   ```bash
   cd lm-evaluation-harness
   ```
2. **Follow the Instructions Within This Directory**

   *Ensure your model is compatible with the evaluation harness requirements.*

## Acknowledgements

- [Karpathy’s nanoGPT](https://github.com/karpathy/nanoGPT) provides the foundational codebase upon which this repo is built.
- [Hugging Face](https://huggingface.co/) for providing the [Fineweb-Edu-100B](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu/) dataset.
- [EleutherAI](https://www.eleuther.ai/) for the [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness).
- [OpenWebText](https://openwebtext2.readthedocs.io/en/latest/) team for replicating the WebText dataset.

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=tensorgi/T6&type=Date)](https://star-history.com/#tensorgi/T6&Date)

## Citation

If you use Tensor Product Attention (TPA) or the Tensor ProducT ATTenTion Transformer (T6) in your research or application, please consider citing it!

```bibtex
@article{zhang2025tensor,
    title={Tensor Product Attention Is All You Need},
    author={Zhang, Yifan and Liu, Yifeng and Yuan, Huizhuo and Qin, Zhen and Yuan, Yang and Gu, Quanquan and Yao, Andrew Chi-Chih},
    journal={arXiv preprint arXiv:2501.06425},
    year={2025},
}
```
