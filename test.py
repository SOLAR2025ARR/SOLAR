import argparse
import re
import copy
from Levenshtein import distance
import pandas as pd
from collections import defaultdict
from tqdm import tqdm
import math
from openai import OpenAI

template = {
    'sys': "You are an expert recommender engine. ",
    'inst': "You need to select a recommendation list considering user's historical interactions. The historical interactions are provided as follows: {history}. The candidate items are:  {candidates}. Please select a recommendation list with {item_count} different items from candidate items."
}

def rm_idx(s):
    s = s.strip()
    return re.sub(r'^(\d+)\. *', '', s, count=1)

def match_idx(s):
    s = s.strip()
    return re.match(r'^(\d+)\. *', s)

def vague_map(titles, all_titles):
    temp = copy.deepcopy(titles)
    for idx, title in enumerate(temp):
        if title in all_titles:
            continue
        for _title in all_titles:
            if distance(str(title), str(_title)) <= 3:
                temp[idx] = _title
                break
    return temp

headers = {"User-Agent": "Test Client"}

def query_vllm_openai(sys, input_text, args):
    for _ in range(args.try_num):
        client = OpenAI(api_key="0",base_url=f"http://0.0.0.0:{args.vllm_port}/v1")
        messages = [{"role": "system", "content": sys},
                    {"role": "user", "content": input_text}]
        result = client.chat.completions.create(messages=messages, model=args.model_name, max_tokens=args.gen_max_length)
        output = result.choices[0].message.content
        if output is None:
            continue
        return output

def process_api_output(raw_output, item_index, topk: int = 10):
    if raw_output[0] == raw_output[-1] == '"' or raw_output[0] == raw_output[-1] == "'":
        raw_output = raw_output[1:-1]
    ts = raw_output.split('\n')
    ts = [rm_idx(_).strip().split('\n')[0].strip() for _ in ts if match_idx(_)]
    ts = [t[1:-1] if t[0] == t[-1] == "'" or t[0] == t[-1] == "\"" else t for t in ts if t != '']
    ts = [t.strip() for t in ts]
    ts = ts[:topk]
    final_ts = vague_map(ts, item_index)
    return final_ts, ts

def get_llama3_templated(template: dict, history, neg_samples, item_count):
    """
    Llama3 chat template
    """
    sys = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>{template['sys']}<|eot_id|>"
    prompt = f"<|start_header_id|>user<|end_header_id|>{template['inst']}<|eot_id|><|start_header_id|>assistant<|end_header_id|>"
    input_text = prompt.format(history=history, item_count=item_count, candidates=neg_samples)
    return sys, input_text

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--testset", type=str, required=True)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--maxlen", type=int, default=10)
    parser.add_argument("--num_neg_samples", type=int, default=19)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--try_num", type=int, default=2, help="The number of attempts to call the API")
    parser.add_argument("--gen_max_length", type=int, default=1024)
    parser.add_argument("--vllm_port", type=int, default=13579)

    args = parser.parse_args()
    titles_ = pd.read_csv(f"{args.data}/titles.csv")
    titles_d = defaultdict(str)
    for row in titles_.itertuples():
        titles_d[int(row.item)] = row.title
    testset = pd.read_pickle(f"{args.testset}.pkl")
    testset = list(zip(testset["history"], testset["candidate"], testset["target"]))
    hr_1, hr_5 = 0, 0
    ndcg_1, ndcg_5 = 0, 0

    miss_count = 0
    valid_count = 0

    for _, (history, candidate, target) in enumerate(tqdm(testset)):
        if type(target) is not str or target.strip() == '':
            continue
        valid_count += 1
        history = ', '.join([f"{idx + 1}. {title}" for idx, title in enumerate(history)])
        neg_samples = '\n'.join([f"{idx + 1}. {title}" for idx, title in enumerate(candidate)])
        sys, input_text = get_llama3_templated(template, history, neg_samples, args.topk)
        output_text = query_vllm_openai(sys, input_text, args)
        titles = list(titles_d.values())
        result, _ = process_api_output(output_text, titles, args.topk)
        if len(result) < args.topk:
            miss_count += 1
        if target not in result:
            continue
        rank = result.index(target) + 1
        if rank <= 1:
            hr_1 += 1
            ndcg_1 += 1
        if rank <= 5:
            hr_5 += 1
            ndcg_5 += 1 / math.log2(rank + 1)
    hr_1 = hr_1 / valid_count
    ndcg_1 = ndcg_1 / valid_count
    hr_5 = hr_5 / valid_count
    ndcg_5 = ndcg_5 / valid_count
    print(f"HR@1: {hr_1:.4f}, HR@5: {hr_5:.4f}")
    print(f"NDCG@1: {ndcg_1:.4f}, NDCG@5: {ndcg_5:.4f}")
    print(f"valid_count: {valid_count}")
    print(f"miss_count: {miss_count}")

if __name__ == "__main__":
    main()