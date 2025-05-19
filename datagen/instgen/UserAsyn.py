import pandas as pd
import random
from openai import AsyncOpenAI
from pprint import pprint
import asyncio
from tqdm import tqdm
import ast

from CorpusGenML import MovieLensCorpus
from CorpusGenBookwoReview import BooksCorpuswoReview
from CorpusGenMovieswoReview import MoviesTVCorpuswoReview

title_path = "<title_path>"
result_path = "<sasrec_inferenc_result_path>"

item_title_df = pd.read_csv(title_path)

item_to_title = dict(zip(item_title_df['item'], item_title_df['title']))

def convert_indices_to_titles(indices, n, mapping, item_review_df):
    selected_indices = indices[-n:]

    title_list = []

    for item_id in selected_indices:
        if item_id in mapping:
            title = mapping[item_id]
            title_list.append(title)

    return title_list


async def gen(profile_template, history_titles, client):
    messages = []

    userinput = {"role": "user", "content": ""}
    userinput["content"] = profile_template.format(interaction=history_titles,
                                                   constraint=MoviesTVCorpuswoReview.constraint[0])
    messages.append(userinput)

    response = await client.chat.completions.create(
        model="deepseek-chat",
        messages=messages,
        stream=False,
        temperature=1.5
    )

    return response.choices[0].message.content


async def process_user(row, client, item_to_title, pbar):
    try:
        user_index = row['user']
        history = row['history']
        topk_items = row['candidates']
        target = row['target']

        n = 5
        history_titles = convert_indices_to_titles(history[:n], n, item_to_title, None)

        iPreference_random_choice = random.choice(MoviesTVCorpuswoReview.Preference['iPreference'])
        ePreference_random_choice = random.choice(MoviesTVCorpuswoReview.Preference['ePreference'])
        nPreference_random_choice = random.choice(MoviesTVCorpuswoReview.Preference['nPreference'])

        vIntention_random_choice = random.choice(MoviesTVCorpuswoReview.Intention['vIntention'])
        sIntention_random_choice = random.choice(MoviesTVCorpuswoReview.Intention['sIntention'])
        eIntention_random_choice = random.choice(MoviesTVCorpuswoReview.Intention['eIntention'])

        results = await asyncio.gather(
            gen(iPreference_random_choice, history_titles, client),
            gen(ePreference_random_choice, history_titles, client),
            gen(nPreference_random_choice, history_titles, client),
            gen(vIntention_random_choice, history_titles, client),
            gen(sIntention_random_choice, history_titles, client),
            gen(eIntention_random_choice, history_titles, client),
            return_exceptions=True
        )

        
        valid_responses = [response for response in results if not isinstance(response, Exception)]

        if len(valid_responses) == len(results):
            pbar.update(1)
            return [user_index] + valid_responses

    except Exception as e:
        print(f"Error processing user {row['user']}: {e}")
        pbar.update(1)
        return None


async def main():
    Key = "<api_key>"

    client = AsyncOpenAI(api_key=Key, base_url="https://api.deepseek.com")

    df = pd.read_csv(result_path)

    df['history'] = df['history'].apply(ast.literal_eval)
    df['candidates'] = df['candidates'].apply(ast.literal_eval)
    df['reranked'] = df['reranked'].apply(ast.literal_eval)

    total_lines = len(df)
    tasks = []

    with tqdm(total=total_lines, desc="Processing users", unit="user") as pbar:
        for _, row in df.iterrows():
            task = process_user(row, client, item_to_title, pbar)
            tasks.append(task)

        results = await asyncio.gather(*tasks)

    valid_results = [res for res in results if res is not None]

    output_df = pd.DataFrame(valid_results, columns=[
        'user', 'iPreference', 'ePreference', 'nPreference', 'vIntention', 'sIntention', 'eIntention'
    ])
    output_df.to_csv('MoviesTVRank_with_user.csv', index=False)


if __name__ == "__main__":
    asyncio.run(main())
