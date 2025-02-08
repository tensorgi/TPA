change 732th line in lm-evaluation-harness/lm_eval/models/huggingface.py for your path to model
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    "gpt2",
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )
                to
                self.tokenizer = transformers.AutoTokenizer.from_pretrained(
                    "YOUR_PATH_TO_MODEL",
                    revision=revision,
                    trust_remote_code=trust_remote_code,
                    use_fast=use_fast_tokenizer,
                )
                if gpt2 tokenizer is not available

convert_to_hf_model.py可以把我们训的模型转换成huggingface模型用于测试，lm-evaluation-harness是我改过的测试框架，里面包含了nanoGPT自定义的GPT模型。

Use example: lm_eval --model hf --model_args pretrained=/data/adorm/hf_output --trust_remote_code --tasks hellaswag --batch_size 16

Use without web access:
HF_ENDPOINT="http://invalid-url.com" lm_eval --model hf --model_args pretrained=/data/adorm/ckpt.pt --trust_remote_code --tasks hellaswag --batch_size 16