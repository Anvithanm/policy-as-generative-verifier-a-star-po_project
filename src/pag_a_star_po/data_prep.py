import json
import pickle
from tqdm import tqdm


def prepare_training_data(pkl_path: str, math_train_dataset, output_path: str = "training_data_with_vstar.json") -> int:
    with open(pkl_path, 'rb') as f:
        offline_dataset = pickle.load(f)
    problems_data = {}
    for item in tqdm(offline_dataset, desc="Grouping by problem"):
        problem = item['problem']
        if problem not in problems_data:
            problems_data[problem] = {
                'v_star': item['v_star'],
                'avg_reward': item['reward'],
                'count': 1
            }
        else:
            data = problems_data[problem]
            data['avg_reward'] = (data['avg_reward'] * data['count'] + item['reward']) / (data['count'] + 1)
            data['count'] += 1
    # build lookup from provided math_train_dataset (expected to be a datasets.Dataset or list-like)
    math_lookup = {}
    for item in math_train_dataset:
        math_lookup[item['problem']] = {'solution': item['solution']}
    training_data = []
    problems_found = 0
    problems_missing = 0
    for problem, data in tqdm(problems_data.items(), desc="Merging"):
        if problem in math_lookup:
            training_data.append({
                'problem': problem,
                'solution': math_lookup[problem]['solution'],
                'v_star': data['v_star'],
                'avg_reward': data['avg_reward'],
                'sample_count': data['count']
            })
            problems_found += 1
        else:
            problems_missing += 1
    with open(output_path, 'w') as f:
        json.dump(training_data, f, indent=2)
    return len(training_data)


def verify_data_format(data_path: str = "training_data_with_vstar.json") -> bool:
    with open(data_path, 'r') as f:
        data = json.load(f)
    required_fields = ['problem', 'solution', 'v_star']
    missing_count = 0
    for i, item in enumerate(data):
        for field in required_fields:
            if field not in item:
                print(f"ERROR: Sample {i} missing field '{field}'")
                missing_count += 1
    if missing_count == 0:
        print("All samples have required fields")
        return True
    return False
