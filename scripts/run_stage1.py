"""Run Stage 1: offline V* estimation (simple runner)
Usage:
    python -m scripts.run_stage1
"""
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from pag_a_star_po.helpers import generate_text, score_solution


def main():
    MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, device_map="auto")
    model.eval()

    train_dataset = load_dataset("qwedsacf/competition_math")['train']
    # Use small subset for quick runs
    subset = train_dataset.select(range(min(100, len(train_dataset))))

    # Example: generate one sample per problem
    for i, prob in enumerate(subset):
        prompt = f"Solve the following math problem step by step.\nProblem: {prob['problem']}\nSolution:"
        gen = generate_text(prompt, model, tokenizer, device=DEVICE, max_new_tokens=256)
        reward = score_solution({'solution': prob['solution']}, gen)
        print(f"Problem {i}: reward={reward} gen[:120]={gen[:120]!r}")

if __name__ == '__main__':
    main()
