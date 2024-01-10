import time
import torch
import csv
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
from transformers.generation import GenerationConfig
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from transformers.trainer_utils import set_seed
from tqdm import tqdm
from torch.utils.data import DataLoader
from memory_profiler import profile
from auto_gptq import AutoGPTQForCausalLM
import sys


quant_type = "bf16"  # quant_type = 'bf16' # bf16, int8, nf4, int4

path_qwen = "/workspace/Qwen-7B-Chat"
path_qwen_int4 = "/workspace/Qwen-7B-Chat-Int4"
use_flash_attn = True


if quant_type == "nf4":
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,  # bnb_4bit_use_double_quant=True,
    )
elif quant_type == "int8":
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
else:
    quantization_config = None


def load_qwen():
    tokenizer = AutoTokenizer.from_pretrained(path_qwen, trust_remote_code=True)
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    # tokenizer.pad_token = '[SEP]'
    model = AutoModelForCausalLM.from_pretrained(
        path_qwen,
        device_map="auto",
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
        quantization_config=quantization_config,
    ).eval()
    config = GenerationConfig.from_pretrained(
        path_qwen, trust_remote_code=True
    )
    model.generation_config = config
      # 可指定不同的生成长度、top_p等相关超参
    return tokenizer, model, config


def load_qwen_int4():
    # Note: The default behavior now has injection attack prevention off.
    tokenizer = AutoTokenizer.from_pretrained(path_qwen_int4, trust_remote_code=True)

    model = AutoGPTQForCausalLM.from_quantized(
        path_qwen_int4,
        device_map="auto",
        use_flash_attn=use_flash_attn,
        trust_remote_code=True,
        use_safetensors=True,
    ).eval()

    # Specify hyperparameters for generation
    config = GenerationConfig.from_pretrained(
        path_qwen_int4, trust_remote_code=True
    )  # 可指定不同的生成长度、top_p等相关超参
    return tokenizer, model



def get_loaded_model():
    tokenizer, model, config = load_qwen()
    return tokenizer, model, config



def generate_output(instruction, tokenizer, model, config):
    pred, his = model.chat(
        tokenizer, instruction, history=None, generation_config=config
    )
    return pred
