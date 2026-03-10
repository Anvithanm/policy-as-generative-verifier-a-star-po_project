import re
import numpy as np
import sympy
from sympy.parsing.latex import parse_latex
from typing import List, Tuple, Optional


def extract_final_answer(solution_text: str) -> Optional[str]:
    if not solution_text:
        return None
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, solution_text)
    if matches:
        return matches[-1].strip()
    return None


def normalize_answer(answer_str):
    if answer_str is None:
        return None
    answer_str = str(answer_str).strip()
    answer_str = re.sub(r'^\$+|\$+$', '', answer_str)
    answer_str = answer_str.replace(' ', '')
    answer_str = re.sub(r'\\text\{([^}]+)\}', r'\1', answer_str)
    answer_str = answer_str.replace('\\dfrac', '\\frac')
    answer_str = answer_str.replace('\\tfrac', '\\frac')
    answer_str = answer_str.replace('\\left', '')
    answer_str = answer_str.replace('\\right', '')

    try:
        num_str = answer_str.replace(',', '')
        if re.match(r'^-?\d+\.?\d*$', num_str):
            return float(num_str)
        if re.match(r'^-?\d+/-?\d+$', num_str):
            parts = num_str.split('/')
            return float(parts[0]) / float(parts[1])
    except Exception:
        pass

    try:
        latex_str = answer_str
        latex_str = latex_str.replace('\\%', '/100')
        expr = parse_latex(latex_str)
        simplified = sympy.simplify(expr)
        if simplified.is_number:
            return float(simplified.evalf())
        return str(simplified)
    except Exception:
        pass

    answer_str = answer_str.lower()
    answer_str = re.sub(r'\\+', '', answer_str)
    return answer_str


def answers_equivalent(ans1, ans2, tolerance=1e-6) -> bool:
    if ans1 is None or ans2 is None:
        return False
    if isinstance(ans1, (int, float)) and isinstance(ans2, (int, float)):
        return abs(float(ans1) - float(ans2)) < tolerance
    if isinstance(ans1, (int, float)) and isinstance(ans2, str):
        try:
            ans2_num = float(ans2)
            return abs(float(ans1) - ans2_num) < tolerance
        except Exception:
            pass
    if isinstance(ans2, (int, float)) and isinstance(ans1, str):
        try:
            ans1_num = float(ans1)
            return abs(ans1_num - float(ans2)) < tolerance
        except Exception:
            pass
    try:
        expr1 = sympy.sympify(str(ans1))
        expr2 = sympy.sympify(str(ans2))
        difference = sympy.simplify(expr1 - expr2)
        if difference == 0:
            return True
        if expr1.is_number and expr2.is_number:
            val1 = float(expr1.evalf())
            val2 = float(expr2.evalf())
            return abs(val1 - val2) < tolerance
    except Exception:
        pass
    str1 = str(ans1).strip().lower()
    str2 = str(ans2).strip().lower()
    return str1 == str2


def score_solution(problem_obj: dict, generated_solution: str, verbose: bool = False) -> float:
    try:
        generated_answer = extract_final_answer(generated_solution)
        ground_truth_answer = extract_final_answer(problem_obj.get('solution'))
        gen_normalized = normalize_answer(generated_answer)
        gt_normalized = normalize_answer(ground_truth_answer)
        is_correct = answers_equivalent(gen_normalized, gt_normalized)
        return 1.0 if is_correct else 0.0
    except Exception as e:
        if verbose:
            print(f"Error in scoring: {e}")
        return 0.0


def evaluate_batch(problems: List[dict], solutions: List[str], verbose: bool = False) -> Tuple[float, List[float]]:
    scores = []
    for prob, sol in zip(problems, solutions):
        score = score_solution(prob, sol, verbose=verbose)
        scores.append(score)
    accuracy = sum(scores) / len(scores) if scores else 0.0
    return accuracy, scores


def generate_text(prompt: str, model, tokenizer, device: str = "cpu", max_new_tokens: int = 512, do_sample: bool = True, top_p: float = 0.95, temp: float = 0.9) -> str:
    try:
        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with __import__('torch').no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temp,
                pad_token_id=tokenizer.eos_token_id
            )
        text = tokenizer.decode(out[0], skip_special_tokens=True)
        generated_part = text[len(prompt):].strip()
        return generated_part
    except Exception as e:
        print(f"Error in generation: {e}")
        return ""
