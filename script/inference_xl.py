import os
import sys
sys.path.append('../')

from diffusers import DPMSolverMultistepScheduler, DDIMScheduler
from Diffusion_xl import DiffusionXLPipeline
import torch
import argparse
import json
import time

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="sdxl", type=str)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="images/")
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    return parser.parse_args()

def sanitize(fn: str) -> str:
    return (fn.strip()
              .replace("/", "_").replace("\\", "_")
              .replace(":", "_").replace("*", "_")
              .replace("?", "_").replace('"', "_")
              .replace("<", "_").replace(">", "_")
              .replace("|", "_"))

def main():
    args = parse_args()
    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')

    with open(args.test_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    items = list(data.items())

    base = os.path.join(args.out_path, f"Ours-{args.model}",
                        os.path.splitext(os.path.basename(args.test_file))[0])
    os.makedirs(base, exist_ok=True)

    if args.model == 'sdxl':
        pipe = DiffusionXLPipeline.from_pretrained(
            "stabilityai/stable-diffusion-xl-base-1.0",
            torch_dtype=torch.float16, use_safetensors=True, variant="fp16"
        )
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config, use_karras_sigmas=True
        )

    pipe = pipe.to(device)
    pipe.prepare_candidates(offline_file="your path of candidates.txt")

    counter = 0
    for bid, (prompt_str, info) in enumerate(items):
        safe_prompt = sanitize(prompt_str)

        Alternating_input = {
            "Bias_level": info["Bias_level"],
            "Alternating_prompt": info["Alternating_prompt"]
        }
        for j in range(10):
            seed = 14273 + bid * 999 + j * 37
            torch.cuda.synchronize()
            t0 = time.time()
            image = pipe(
                Alternating_prompts=Alternating_input,
                batch_size=1,
                num_inference_steps=args.num_inference_steps,
                height=args.height,
                width=args.width,
                seed=seed
            ).images[0]
            torch.cuda.synchronize()
            t1 = time.time()
            fname = f"{counter}_{safe_prompt}.jpg"
            image.save(os.path.join(base, fname))
            counter += 1
            print(f"[{prompt_str}] ({j+1}/10) seed={seed} time={t1-t0:.2f}s")

if __name__ == "__main__":
    main()