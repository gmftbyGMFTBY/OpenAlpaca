## Fine-tuning

We fine-tune OpenLLaMa-7B base model on the dolly dataset with the following hyper-parameters:

| Hyperparameter | LLaMa-7B |
| -------------- | -------- |
| Batch size     | 64 |
| Learning rate  | 2e-5 |
| Epochs         | 2 |
| Max length     | 1024 |

### 1. Prepare the environments

The fine-tuning runs on the 8 A800 80G GPU (cuda 11.6) server with conda.

Firstly, prepare the conda environment for running the fine-tuning

```bash
pip install -r requirements.txt
```

If you meet the installing errors about torch, please install torch by following commands manually:

```bash
pip install torch==1.13.1+cu116 -f https://download.pytorch.org/whl/torch/
```

Finally, please download the OpenLLaMa model by [this link](https://huggingface.co/openlm-research/open_llama_7b_preview_200bt)

### 2. Fine-tune

Before fine-tuning, you should check the deepspeed configuration in `src/dsconfig/openllama_stage_1.json`. In this work, we train the OpenLlama model with DeepSpeed ZeRO-2 on 8 A100 GPUs. Change the `train_batch_size`, `train_micro_batch_sie_per_gpu`.

Then, just run the following commands:

```bash
./scripts/train_sft.sh
```
This command will train the openllama model (saved under `/home/johnlan/pretrained_models/openllama`) with `../data/openalpaca.json` dataset.

After fine-tuning, you could find the saved checkpoint under `--save_path`, containing the tokenizer, configuration, and deepspeed checkpoints.
Please running the following codes to convert the deepspeed checkpoints to torch models:

```bash
python zero_to_fp32.py . pytorch_model.bin
```

Then, you could get the fine-tuned checkpoints (`--save_path`)

### 3. Test the model

Test the model by running the following command:

```bash
./scripts/inference.sh
```
