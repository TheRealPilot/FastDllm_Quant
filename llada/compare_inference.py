import torch
import time
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from .model.modeling_llada import LLaDAModelLM
from datasets import load_dataset
from tqdm import tqdm
import gc

dataset = load_dataset("openai/gsm8k", "main", split="test[:50]")
prompts = []
for item in dataset:
    prompts.append(item["question"])


# ========== AUTOREGRESSIVE (Transformers) ==========
def benchmark_autoregressive(model, tokenizer, prompts, max_new_tokens, device, batch_size=1):
    model.eval()

    decode_times = []
    throughputs = []
    total_tokens = 0
    total_time = 0

    # Prepare all prompts
    all_formatted = []
    for prompt in prompts:
        p = [{"role": "user", "content": prompt}, ]
        formatted = tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
        all_formatted.append(formatted)

    # Process in batches
    for batch_start in tqdm(range(0, len(all_formatted), batch_size), desc="Autoregressive Inference"):
        batch_end = min(batch_start + batch_size, len(all_formatted))
        batch_prompts = all_formatted[batch_start:batch_end]

        inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True).to(device)
        input_len = inputs.input_ids.shape[1]

        with torch.inference_mode():
            s = time.perf_counter()
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,
            )
            decode_time = time.perf_counter() - s

            # Calculate actual tokens generated (excluding padding)
            batch_tokens = 0
            for i in range(outputs.shape[0]):
                # Count non-pad tokens in output beyond input length
                output_tokens = outputs[i, input_len:]
                non_pad = (output_tokens != tokenizer.pad_token_id).sum().item()
                batch_tokens += non_pad

            decode_times.append(decode_time)
            throughputs.append(batch_tokens / decode_time)
            total_tokens += batch_tokens
            total_time += decode_time

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {
        "throughputs": throughputs,
        "decode_times": decode_times,
        "total_tokens": total_tokens,
        "total_time": total_time,
    }

# ========== DIFFUSION (Fast-dLLM) ==========
def benchmark_diffusion(model, tokenizer, prompts, gen_length, steps, block_length, device, batch_size=1):
    from .generate import generate_with_dual_cache, generate_with_prefix_cache, generate

    decode_times = []
    total_nfe = []
    throughputs = []
    total_tokens = 0
    total_time = 0

    # Prepare all prompts
    all_input_ids = []
    for prompt in prompts:
        p = [{"role": "user", "content": prompt}, ]
        formatted = tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
        input_ids = tokenizer(formatted)['input_ids']
        all_input_ids.append(input_ids)

    # Process in batches
    for batch_start in tqdm(range(0, len(all_input_ids), batch_size), desc="Diffusion Inference"):
        batch_end = min(batch_start + batch_size, len(all_input_ids))
        batch_inputs = all_input_ids[batch_start:batch_end]

        # Pad to same length within batch
        max_len = max(len(ids) for ids in batch_inputs)
        padded_inputs = []
        pad_lens = []
        for ids in batch_inputs:
            pad_len = max_len - len(ids)
            pad_lens.append(pad_len)
            # Left-pad with pad token
            padded = [tokenizer.pad_token_id] * pad_len + ids
            padded_inputs.append(padded)

        input_ids = torch.tensor(padded_inputs, dtype=torch.long, device=device)

        # Build attention mask for batched inference (if batch_size > 1)
        if batch_size > 1:
            attention_mask = torch.zeros(
                (input_ids.shape[0], 1, max_len + gen_length, max_len + gen_length),
                device=device, dtype=torch.bool
            )
            for i, pad_len in enumerate(pad_lens):
                attention_mask[i, :, pad_len:, pad_len:] = True

        with torch.inference_mode():
            s = time.perf_counter()
            outputs, nfe = generate(
                model, input_ids,
                steps=steps, gen_length=gen_length, block_length=block_length,
                temperature=0., remasking='low_confidence'
            )
            decode_time = time.perf_counter() - s

            # Calculate tokens generated (excluding padding)
            batch_tokens = 0
            for i in range(len(batch_inputs)):
                batch_tokens += outputs.shape[1] - len(batch_inputs[i])

            decode_times.append(decode_time)
            throughputs.append(batch_tokens / decode_time)
            total_nfe.append(nfe)
            total_tokens += batch_tokens
            total_time += decode_time

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    return {
        "throughputs": throughputs,
        "decode_times": decode_times,
        "total_nfe": total_nfe,
        "total_tokens": total_tokens,
        "total_time": total_time,
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
    parser.add_argument("--gen_length", type=int, default=256, help="Generation length")
    parser.add_argument("--steps", type=int, default=128, help="Diffusion steps")
    parser.add_argument("--block_length", type=int, default=256, help="Block length for parallel generation")
    args = parser.parse_args()

    device = "cuda:0"

    # Load AR model (Qwen)
    ar_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    ar_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen3-8B",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    # Load LLaDA with flash attention enabled (matching eval_llada.py)
    llada_tokenizer = AutoTokenizer.from_pretrained('GSAI-ML/LLaDA-8B-Instruct', trust_remote_code=True)
    llada_config = AutoConfig.from_pretrained('GSAI-ML/LLaDA-8B-Instruct')
    llada_config.flash_attention = True
    llada_model = LLaDAModelLM.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        config=llada_config,
        device_map="auto"
    )
    llada_model.eval()

    print(f"\n{'='*60}")
    print(f"Configuration: batch_size={args.batch_size}, gen_length={args.gen_length}, steps={args.steps}, block_length={args.block_length}")
    print(f"{'='*60}\n")

    # Run benchmarks with matching batch sizes
    llada_perf = benchmark_diffusion(
        llada_model, llada_tokenizer, prompts,
        gen_length=args.gen_length, steps=args.steps, block_length=args.block_length,
        device=device, batch_size=args.batch_size
    )
    ar_perf = benchmark_autoregressive(
        ar_model, ar_tokenizer, prompts,
        max_new_tokens=args.gen_length, device=device, batch_size=args.batch_size
    )

    # Print results
    print(f"\n{'='*60}")
    print("RESULTS")
    print(f"{'='*60}")
    print(f"\nLLaDA (Diffusion):")
    print(f"  Average batch throughput: {sum(llada_perf['throughputs'])/len(llada_perf['throughputs']):.2f} tokens/sec")
    print(f"  Total throughput: {llada_perf['total_tokens']/llada_perf['total_time']:.2f} tokens/sec")
    print(f"  Total tokens: {llada_perf['total_tokens']}, Total time: {llada_perf['total_time']:.2f}s")
    print(f"  Average NFE: {sum(llada_perf['total_nfe'])/len(llada_perf['total_nfe']):.2f}")

    print(f"\nQwen (Autoregressive):")
    print(f"  Average batch throughput: {sum(ar_perf['throughputs'])/len(ar_perf['throughputs']):.2f} tokens/sec")
    print(f"  Total throughput: {ar_perf['total_tokens']/ar_perf['total_time']:.2f} tokens/sec")
    print(f"  Total tokens: {ar_perf['total_tokens']}, Total time: {ar_perf['total_time']:.2f}s")
    print(f"{'='*60}\n")

if __name__=="__main__":
    main()