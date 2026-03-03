import pandas as pd
import time
from tqdm import tqdm
import openai
import anthropic
import google.generativeai as genai


# API key
openai.api_key = ""
claude_client = anthropic.Anthropic(api_key="")
genai.configure(api_key="")


df = pd.read_csv("file.csv")


def make_prompt(preceding_discourse, target_sentence):
    return (
        f"Read the given context and the following sentence, and rate how naturally the sentence follows the context.\n\n"
        f"Context: {preceding_discourse}\n"
        f"Sentence: {target_sentence}\n"
        f"Please respond with only the number between 1 and 5 (1 = Not natural at all, 5 = Completely natural).\n"
        f"Score:"
    )


def query_gpt(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response['choices'][0]['message']['content']

def query_claude(prompt):
    response = claude_client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=30,
        temperature=0,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text

def query_gemini(prompt):
    model = genai.GenerativeModel('gemini-1.5-flash')
    response = model.generate_content(prompt)
    return response.text


n_runs = 10
results = []


for idx, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating rows"):
    prompt = make_prompt(row["preceding_discourse"], row["target_sentence"])

    for model_name, query_func in {
        # "gpt": query_gpt,
        "claude": query_claude
        # "gemini": query_gemini
    }.items():
        for run in range(1, n_runs + 1):
            try:
                response = query_func(prompt)
                score = next((c for c in response if c in ['1', '2', '3', '4', '5']), "")
            except Exception as e:
                print(f"[{model_name}] Error on row {idx}, run {run}: {e}")
                score = ""

            results.append({
                "original_index": idx + 1,
                "run": run,
                "model": model_name,
                "preceding_discourse": row["preceding_discourse"],
                "target_sentence": row["target_sentence"],
                "relation_type": row["relation_type"],
                "word_order": row["word_order"],
                "is_type": row["is_type"],
                "score": score
            })

            time.sleep(0.3)


df_results = pd.DataFrame(results)
df_results.to_csv("save_results.csv", index=False)
