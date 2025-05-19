import numpy as np
import pandas as pd
from openai import AsyncOpenAI
import asyncio
from typing import List, Tuple
import re
import time
import argparse
from collections import defaultdict
import logging
import os

class PromptCollection:
    system_prompt: str = """
    You excel at role-playing. Picture yourself as a user exploring a {0} recommendation system. Your goal is to understand and evaluate how surprising and delightful the {0} recommendations are based on your viewing history.
    """

    understand_serendipity: str = """
    You are an avid {0} enthusiast, currently watching a {0}.

    You are about to receive a list of {0} recommendations for you. You might recognize some of these {0} because your knowledge base might include their content.

    Your goal is to understand and evaluate how surprising and delightful these {0} recommendations are.

    Before that, you should have an understanding of what is serendipity.

    Based on your past knowledge, give your understanding of serendipity.

    Expected Output:

    Detailed Analysis: Your analysis on what is serendipity, what it takes for a {0} to be serendipitous to an user, how to identify whether a movie is serendipitous.

    Clear and Structured Insights: Ensure your analysis is thorough and easy to understand.
    """

    refined_understanding: str = """
    Great job on your initial analysis! Based on your detailed insights, we've developed a scoring method to help quantify your serendipity evaluations. I've mapped your descriptive analysis into a five-point integer scale (1-5) and made some adjustments to align with this rating system.

    Here's what you'll do next:
    1. Learn the Scoring Standard: Take a look at the serendipity ratings provided for each {0}. These scores reflect how surprising and delightful each recommendation was, according to your analysis.
      Consider the user's opinion to the following questions:
          "I was (or, would have been) surprised that MovieLens picked this {0} to recommend to me."
          "This is the type of movie I would not normally discover on my own; I need a recommender system to find {0} like this one."
          "Watching this movie broadened my preferences. Now I am interested in a wider selection of {0}."
      Serendipity Rating Metric:
        1 - Strongly Disagree
        2 - Disagree
        3 - Neither Agree nor Disagree
        4 - Agree
        5 - Strongly Agree

    2. Refine Your Analysis: Adjust your analysis if necessary. Use the scores as a guideline to refine your understanding of serendipity.

    Expected Output:
    Refined Analysis: Update your report to include insights from the serendipity scoring standard, comparing them with your initial understanding.

    Aligned Insights: Make sure your final analysis aligns with your true tastes, providing a clearer understanding of what makes recommendations serendipitous for you.
    """

    rerank_inst: str = """
    Now that you have understand what is serendipity in recommendation system. Based on your understanding, given a user's historical watching history, you are asked to select {k} most serendipitous movies from a candidate list.

    The user's watching history in chronological order is as follows:
    {interaction_history}

    The candidate list is as follows:
    {candidate_list}

    You may consider the following workflow:

    1. Summarize the preference from the user's watching history with a short profile description and analysis.

    2. Select movies from the candidate list based on the user's preference and your understanding of serendipity.

    Expected Output:
    Recommendation List: {k} most serendipitous movies from the candidate list, represented by their IDs.

    Your answer should be structured like this: [ID1, ID2, ...], where IDs should be digits.
    """

    rerank_inst_icl: str = """
    Now that you have understood what serendipity is in recommendation systems, you are asked to select {k} most serendipitous movies from a candidate list based on a user's historical watching history.

    Additionally, to help guide your decision-making, I will provide you with a few examples of other users. For each example, I will show you the user's history, the item they liked (positive example), and the item they did not like (negative example). Learn from these examples to better understand how to differentiate between positive and negative recommendations.

    Below are some examples for you to learn from:
    {icl_examples}

    Based on these examples and your understanding of serendipity, now analyze the new user's history and candidate list.

    The new user's watching history in chronological order is as follows:
    {interaction_history}

    This gives us insight into what kinds of movies the user tends to like and dislike. Consider this example as you move forward.

    The candidate list is as follows:
    {candidate_list}

    You may consider the following workflow:

    1. Summarize the preference from the new user's watching history with a short profile description and analysis.

    2. Select movies from the candidate list based on the user's preference, your understanding of serendipity, and what you learned from the provided examples.

    Expected Output:
    Recommendation List: {k} most serendipitous movies from the candidate list, represented by their IDs.

    Your answer should be structured like this: [ID1, ID2, ...], where IDs should be digits.
    """
    icl_sample_prompt: str = """
    This user has watched the following movies:
    {icl_user_history}

    Based on their watching history, they particularly enjoyed the movie:
    {icl_positive_item}

    However, they did not enjoy the movie:
    {icl_negative_item}


    """



def str_tuple_list(l: List[Tuple[int, str]]) -> str:
    l = [f"(ID: {t[0]}, Title: {t[1]})" for t in l]
    l = ', '.join(l)
    l = '[' + l + ']'
    return l

class LLMReranker:
    def __init__(self, api_key: str, model: str, base_url: str, topic: str):
        self.model = model
        self.client = AsyncOpenAI(api_key=api_key, base_url=base_url)
        PromptCollection.system_prompt = PromptCollection.system_prompt.format(topic)
        PromptCollection.understand_serendipity = PromptCollection.understand_serendipity.format(topic)
        PromptCollection.refined_understanding = PromptCollection.refined_understanding.format(topic)
        # log the prompt collection content into a log file
        # create logs directory if it does not exist
        os.makedirs('logs', exist_ok=True)
        with open(f"logs/{model}-{topic}.log", 'w') as f:
            pass
        logging.basicConfig(filename=f"logs/{model}-{topic}.log", level=logging.INFO)
        logging.info(PromptCollection.system_prompt)
        logging.info(PromptCollection.understand_serendipity)
        logging.info(PromptCollection.refined_understanding)
        logging.info(PromptCollection.rerank_inst)



    async def _async_request(self, client, messages):
        response = await client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content

    def _extract_last_ids(self, text: str):
        pattern = r"\[([0-9,\s]+)\]"
        matches = re.findall(pattern, text)
        if matches:
            last_match = matches[-1]
            ids_list = [int(x.strip()) for x in last_match.split(',')]
            return ids_list
        else:
            return []

    async def _async_rerank_step(self, history: List[Tuple[int, str]], candidates: List[Tuple[int, str]], topk: int, history_max_len: int = 10) -> List[int]:
        if len(history) > history_max_len:
            history = history[-history_max_len:]

        system_prompt = PromptCollection.system_prompt

        guess_step = PromptCollection.understand_serendipity

        refined_step = PromptCollection.refined_understanding

        predict_step = PromptCollection.rerank_inst.format(
            interaction_history=str_tuple_list(history),
            candidate_list=str_tuple_list(candidates),
            k = topk
        )

        system_prompt = {"role": "system", "content": system_prompt}
        step1 = {"role": "user", "content": guess_step}
        step2 = {"role": "user", "content": refined_step}
        step3 = {"role": "user", "content": predict_step}

        messages = []

        messages.append(system_prompt)
        messages.append(step1)

        response = await self._async_request(self.client, messages)
        messages.append({"role": "assistant", "content": response})

        messages.append(step2)
        response = await self._async_request(self.client, messages)
        messages.append({"role": "assistant", "content": response})

        messages.append(step3)
        response = await self._async_request(self.client, messages)

        return self._extract_last_ids(response)
    
    async def _async_rerank_step_icl(self, dataset: Tuple[List[List[Tuple[int, str]]], List[List[Tuple[int, str]]], List[Tuple[int, str]]], history: List[Tuple[int, str]], candidates: List[Tuple[int, str]], topk: int, history_max_len: int = 10, num_icls: int = 3) -> List[int]:
        if len(history) > history_max_len:
            history = history[-history_max_len:]

        histories, candidate_lists, target_items = dataset

        all_indices = list(range(len(histories)))

        icl_sample_indices = np.random.choice(all_indices, num_icls, replace=False)

        num_negatives = 1  

        icl_examples = []
        for i in icl_sample_indices:
            user_history = histories[i]
            user_candidates = candidate_lists[i]
            positive_item = target_items[i]

            if len(user_history) > history_max_len:
                truncated_history = user_history[-history_max_len:]

            full_history = histories[i]
            remaining_history = [item for item in user_history if item not in truncated_history]

            if remaining_history:
                random_indices = np.random.choice(len(remaining_history), min(num_negatives, len(remaining_history)), replace=False)
                negative_items = [remaining_history[idx] for idx in random_indices]
            else:
                negative_items = [("N/A", "No available negative item")] * num_negatives  # 如果没有负样本可选

            icl_example_text = PromptCollection.icl_sample_prompt.format(
                icl_user_history=str_tuple_list(truncated_history).replace("\\", ""),
                icl_positive_item=str_tuple_list([positive_item]).replace("\\", ""),
                icl_negative_item=str_tuple_list(negative_items).replace("\\", "")
            )


            icl_examples.append(icl_example_text)



        system_prompt = PromptCollection.system_prompt

        guess_step = PromptCollection.understand_serendipity

        refined_step = PromptCollection.refined_understanding

        predict_step = PromptCollection.rerank_inst_icl.format(
            interaction_history=str_tuple_list(history),
            candidate_list=str_tuple_list(candidates),
            k = topk,
            icl_examples=icl_examples
        )

        system_prompt = {"role": "system", "content": system_prompt}
        step1 = {"role": "user", "content": guess_step}
        step2 = {"role": "user", "content": refined_step}
        step3 = {"role": "user", "content": predict_step}

        print(step3)

        messages = []

        messages.append(system_prompt)
        messages.append(step1)

        response = await self._async_request(self.client, messages)
        messages.append({"role": "assistant", "content": response})

        messages.append(step2)
        response = await self._async_request(self.client, messages)
        messages.append({"role": "assistant", "content": response})

        messages.append(step3)
        response = await self._async_request(self.client, messages)

        return self._extract_last_ids(response)
    
    async def _async_llm_rerank(self, dataset: Tuple[List[List[Tuple[int, str]]], List[List[Tuple[int, str]]], List[Tuple[int, str]]], topk: int = 10, maxlen: int = 20, candidate_len: int = 20, limit: int = 10000, start = 0) -> List:
        histories, candidate_lists, target_items = dataset
        result = []
        max_retry_step = 3

        async def process_item(history, candidates, target):
            if len(history) > maxlen:
                history = history[-maxlen:]
            if len(candidates) > candidate_len:
                candidates = candidates[:candidate_len]
            retry_cnt = 0
            while retry_cnt <= max_retry_step:
                try:
                    reranked = await self._async_rerank_step(history, candidates, topk)
                    return (history, candidates, reranked, target)
                except Exception as e:
                    retry_cnt += 1
                    if retry_cnt > max_retry_step:
                        raise e

        # Create a list of tasks
        tasks = [process_item(history, candidates, target) for history, candidates, target in zip(histories[start:limit], candidate_lists[start:limit], target_items[start:limit])]

        # Run all tasks concurrently and get results
        result = await asyncio.gather(*tasks)

        return result

    async def _async_llm_rerank_icl(self, dataset: Tuple[List[List[Tuple[int, str]]], List[List[Tuple[int, str]]], List[Tuple[int, str]]], topk: int = 10, maxlen: int = 20, candidate_len: int = 20, limit: int = 10000, start = 0, num_icls = 2) -> List:
        histories, candidate_lists, target_items = dataset
        result = []
        max_retry_step = 3

        async def process_item(dataset, history, candidates, target, num_icls):
            if len(history) > maxlen:
                history = history[-maxlen:]
            if len(candidates) > candidate_len:
                candidates = candidates[:candidate_len]
            retry_cnt = 0
            while retry_cnt <= max_retry_step:
                try:
                    reranked = await self._async_rerank_step_icl(dataset, history, candidates, topk, num_icls=num_icls)
                    return (history, candidates, reranked, target)
                except Exception as e:
                    retry_cnt += 1
                    if retry_cnt > max_retry_step:
                        raise e

        # Create a list of tasks
        tasks = [process_item(dataset, history, candidates, target, num_icls) for history, candidates, target in zip(histories[start:limit], candidate_lists[start:limit], target_items[start:limit])]

        # Run all tasks concurrently and get results
        result = await asyncio.gather(*tasks)

        return result

    async def rerank(self, dataset: Tuple[List[List[Tuple[int, str]]], List[List[Tuple[int, str]]], List[Tuple[int, str]]], topk: int = 10, maxlen: int = 20, candidate_len: int = 20, limit: int = 10000, start = 0) -> List:
        result = await self._async_llm_rerank(dataset, topk = topk, maxlen = maxlen, candidate_len = candidate_len, limit = limit, start = start)
        result = [([h_[0] for h_ in h], [c_[0] for c_ in c], r, t[0]) for h, c, r, t in result if len(r) > 0]
        return result
    
    async def rerank_icl(self, dataset: Tuple[List[List[Tuple[int, str]]], List[List[Tuple[int, str]]], List[Tuple[int, str]]], topk: int = 10, maxlen: int = 20, candidate_len: int = 20, limit: int = 10000, start = 0, num_icls = 2) -> List:
        result = await self._async_llm_rerank_icl(dataset, topk = topk, maxlen = maxlen, candidate_len = candidate_len, limit = limit, start = start, num_icls=num_icls)
        result = [([h_[0] for h_ in h], [c_[0] for c_ in c], r, t[0]) for h, c, r, t in result if len(r) > 0]
        return result
    

def prepare_dataset(recommendation_result_path: str, title_path: str) -> Tuple[List[List[Tuple[int, str]]], List[List[Tuple[int, str]]], List[Tuple[int, str]]]:
    """
    Read recommendation lists and interaction history from recommendation_result_path and title_path.
    Read item descriptions from the title file.
    Args:
        recommendation_result_path: Path to the recommendation result file.
        title_path: Path to the description file.
    Returns:
        (histories, candidate_lists, target_items)
    """
    titles = pd.read_csv(title_path)
    recommendations = pd.read_csv(recommendation_result_path)
    id2title = defaultdict(str)
    for idx, row in titles.iterrows():
        title = row['title']
        movie_id = int(row['item'])
        id2title[movie_id] = title

    histories, candidate_lists, target_items = [], [], []

    for _, row in recommendations.iterrows():
        history = list(map(int, row['history'].split()))
        topk_items = list(map(int, row['topk_items'].split()))
        target = int(row['target'])
        history = [(idx, id2title[idx]) for idx in history]
        topk_items = [(idx, id2title[idx]) for idx in topk_items]
        target = (target, id2title[target])
        histories.append(history)
        candidate_lists.append(topk_items)
        target_items.append(target)
    return (histories, candidate_lists, target_items)

def sample_from_dataset(dataset: Tuple[List[List[Tuple[int, str]]], List[List[Tuple[int, str]]], List[Tuple[int, str]]], k: int) -> Tuple[List[List[Tuple[int, str]]], List[List[Tuple[int, str]]], List[Tuple[int, str]]]:
    """
    randomly select k samples from the given dataset.
    """
    histories, candidate_lists, target_items = dataset
    # fix random seed 
    np.random.seed(42)
    indices = np.random.choice(len(histories), k, replace=False)
    sampled_histories = [histories[i] for i in indices]
    sampled_candidate_lists = [candidate_lists[i] for i in indices]
    sampled_target_items = [target_items[i] for i in indices]
    return (sampled_histories, sampled_candidate_lists, sampled_target_items)

async def main():
    api_key = "<replace/by/your/api/key>"
    parser = argparse.ArgumentParser()
    parser.add_argument("--topic", type=str, required=True)
    parser.add_argument("--num_samples", type=int, required=True)
    parser.add_argument("--num_icls", type=int, required=True)
    parser.add_argument("--res_path", type=str, required=True)
    parser.add_argument("--title_path", type=str, required=True)
    parser.add_argument("--base_model", type=str, default="")
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--base_url", type=str, default="https://api.deepseek.com")
    parser.add_argument("--chunk_size", type=int, default=500)
    parser.add_argument("--output_file", type=str, default="rerank_result.csv")

    args = parser.parse_args()

    # prepare dataset
    dataset = prepare_dataset(args.res_path, args.title_path)
    # sample from dataset
    if args.num_samples > 0 and args.num_samples < len(dataset[0]): # -1 for using the full dataset
        dataset = sample_from_dataset(dataset, args.num_samples)

    # create LLMReranker
    reranker = LLMReranker(api_key, args.model, args.base_url, args.topic)

    result = []
    chunk_size = args.chunk_size

    start_time = time.time()

    # iterate over the dataset
    for i in range(0, len(dataset[0]), chunk_size):
        start = i
        end = min(i + chunk_size, len(dataset[0]))
        result.extend(await reranker.rerank_icl(dataset, start=start, limit=end, num_icls=args.num_icls))
        current_time = time.time()
        print(f"Processed {end} samples in {current_time - start_time} seconds")

    end_time = time.time()
    print(f"Total time taken: {end_time - start_time} seconds")

    # make a dataframe from the result
    result_df = pd.DataFrame(result, columns=['history', 'candidates', 'topk_items', 'target'])
    # transform list to string, format: id1 id2 id3...
    result_df['history'] = result_df['history'].apply(lambda x: ' '.join([str(t) for t in x]))
    result_df['candidates'] = result_df['candidates'].apply(lambda x: ' '.join([str(t) for t in x]))
    result_df['topk_items'] = result_df['topk_items'].apply(lambda x: ' '.join([str(t) for t in x]))
    # output the result to a csv file
    output_path = args.output_file
    result_df.to_csv(output_path, index=False)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())