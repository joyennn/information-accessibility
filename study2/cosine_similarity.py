import pandas as pd
from sentence_transformers import SentenceTransformer, util
import torch
from tqdm import tqdm

file_path = "file.csv"
df = pd.read_csv(file_path)

model = SentenceTransformer('all-mpnet-base-v2')
tqdm.pandas(desc="Computing cosine similarity")

def compute_similarity(row):
    context = row['preceding_discourse']
    link = row['link']

    try:
        context_emb = model.encode(context, convert_to_tensor=True)
        link_emb = model.encode(link, convert_to_tensor=True)
        similarity = util.cos_sim(link_emb, context_emb)[0][0].item()
    except Exception as e:
        similarity = None
    return similarity

df['cosine_similarity'] = df.apply(compute_similarity, axis=1)

output_path = "save_file.csv"
df.to_csv(output_path, index=False)
