import openai
import pandas as pd
import time

### Task 1: Context-based Generation ###

# API key
openai.api_key = "open-ai-api-key"

df = pd.read_csv("file.csv", header=None)

def build_prompt(sentence):
    return f"""Please generate a short and natural discourse that could come before the following sentence in a conversation or written text:

{sentence}
"""


preceding_discourses = []

sentences = df[0].tolist()

for i, sentence in enumerate(sentences):
    prompt = build_prompt(sentence)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        result = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        result = f"[Error: {e}]"

    preceding_discourses.append(result)
    print(f"{i+1}/{len(sentences)} complete.")
    time.sleep(0.5)  


df['Preceding'] = preceding_discourses

df.to_csv("save_file.csv", index=False, header=False)

print("✅ Complete!")



### Task 2: Construction-based Generation ###

import openai
import pandas as pd
import time

# API key
openai.api_key = "open-ai-api-key"  # 

df = pd.read_csv("file.csv", header=None)

def build_prompt(discourse):
    return f"""Please generate an passive sentence with by-phrase at the end (without any extra expressions), which can naturally follow the given discourse:

{discourse}
"""

### prompts ###
#Please generate a preposing sentence (where an argument appearing to the left of its canonical position like O-S-V), which can naturally follow the given discourse:
#Please generate an inversion sentence (where a prepositional phrase comes first, followed by the verb, and then a noun phrase. like PP-V-NP), which can naturally follow the given discourse:
#Please generate an passive sentence with by-phrase at the end (without any extra expressions), which can naturally follow the given discourse:


prompt = build_prompt(discourse)
response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
result = response['choices'][0]['message']['content'].strip()
print(result)


target_sentence = []

discourses = df[0].tolist()

for i, discourse in enumerate(discourses):
    prompt = build_prompt(discourse)

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
        )
        result = response['choices'][0]['message']['content'].strip()
    except Exception as e:
        result = f"[Error: {e}]"

    target_sentence.append(result)
    print(f"{i+1}/{len(discourses)} complete.")
    time.sleep(0.3) 


df['target_sentence'] = target_sentence

df.to_csv("save_file.csv", index=False, header=False)

print("✅ Complete!")
