from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import numpy as np
from datasets import load_dataset
import torch
import torch.nn as nn
from tqdm import tqdm
import argparse
import json
import os

@torch.no_grad()
def batched_generate_n(model, tokenizer, prompts, n, batch_size=8, max_new_tokens=256, temperature=1, device='cuda'): 
    
    model = model.to(device)
    model.eval()
    
    generated = []
    for i in tqdm(range(0, len(prompts), batch_size)):
        batch = prompts[i:i+batch_size]
        batch = [[{"role": "user", "content": p}] for p in batch]
        inputs = tokenizer.apply_chat_template(batch, tokenize=True, padding=True, return_tensors="pt").to(device)
        attn_mask = torch.ones_like(inputs).to(device)
        attn_mask[inputs == tokenizer.pad_token_id] = 0
        outputs = model.generate(inputs, attention_mask=attn_mask, max_new_tokens=max_new_tokens, num_return_sequences=n, temperature=temperature)
        outputs = outputs[:, inputs.shape[1]+5:]
        outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        for j in range(len(batch)):
            generated.append(outputs[j*n:(j+1)*n])
            
        del inputs, attn_mask, outputs
        torch.cuda.empty_cache()
        
    return generated

def main(args): 
    
    set_seed(args.seed)

    alpaca_eval = load_dataset("tatsu-lab/alpaca_eval", "alpaca_eval")["eval"]
    
    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16)
    model = torch.compile(model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token = tokenizer.eos_token
    
    generated = batched_generate_n(model, tokenizer, alpaca_eval['instruction'], n=args.n, batch_size=args.batch_size, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    
    out = {'instruction': alpaca_eval['instruction'], 'generated': generated}
    with open(f"{args.output}.json", "w") as f:
        json.dump(out, f)
        
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=str, default="data/llama3_512")
    args = parser.parse_args()
    main(args)