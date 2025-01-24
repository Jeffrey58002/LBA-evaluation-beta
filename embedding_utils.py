# embedding_utils.py

import os
from openai import OpenAI
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import math

# 你可能需要設置環境變數或某種機制來放API key
# 這裡簡單示範，如果要保密就不要硬編在程式裡
#openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_API_KEY")
client = OpenAI(api_key="your api key")

def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return np.array(response.data[0].embedding)

def calculate_similarity(text1: str, text2: str) -> float:
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

def compare_as_whole_text(expected_list, actual_list, threshold):
    expected_text = " ".join(expected_list)
    actual_text = " ".join(actual_list)

    similarity = calculate_similarity(expected_text, actual_text)
    pass_check = math.isclose(similarity, threshold, rel_tol=1e-9) or similarity >= threshold

    return similarity, pass_check

# 其他跟 embedding 或 similarity 計算相關的函式，如果有，就一併放在這裡
