from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import math
from itertools import product

# Initialize OpenAI client with API key
client = OpenAI(api_key="")

# Function to get embeddings from OpenAI
def get_embedding(text: str, model: str = "text-embedding-3-small") -> np.ndarray:
    response = client.embeddings.create(
        input=text,
        model=model
    )
    return np.array(response.data[0].embedding)

# Function to calculate similarity
def calculate_similarity(text1: str, text2: str) -> float:
    embedding1 = get_embedding(text1)
    embedding2 = get_embedding(text2)
    similarity = cosine_similarity([embedding1], [embedding2])[0][0]
    return similarity

def compare_similarity(expected_list, actual_list, threshold):
    n_expected = len(expected_list)
    n_actual = len(actual_list)
    similarity_matrix = np.zeros((n_expected, n_actual))

    # 計算相似度矩陣
    for i, j in product(range(n_expected), range(n_actual)):
        similarity_matrix[i, j] = calculate_similarity(expected_list[i], actual_list[j])

    # 找到對應每個 e_i 的最佳匹配相似度
    best_matches = similarity_matrix.max(axis=1)
    avg_similarity = np.mean(best_matches)

    # 是否通過檢查 (加入 math.isclose)
    pass_check = math.isclose(avg_similarity, threshold, rel_tol=1e-9) or avg_similarity >= threshold
    return avg_similarity, pass_check

def compare_as_whole_text(expected_list, actual_list, threshold):
    # 將列表合併為單一文本
    expected_text = " ".join(expected_list)
    actual_text = " ".join(actual_list)

    # 計算相似度
    similarity = calculate_similarity(expected_text, actual_text)

    # 是否通過檢查
    pass_check = math.isclose(similarity, threshold, rel_tol=1e-9) or similarity >= threshold
    return similarity, pass_check


def hybrid_similarity(expected_list, actual_list, thresholds):
    list_similarity, list_pass = compare_similarity(
        expected_list,
        actual_list,
        thresholds[0]
    )
    whole_similarity, whole_pass = compare_as_whole_text(
        expected_list,
        actual_list,
        thresholds[1]
    )

    # 加權平均 (此處示範 1:1 權重)
    final_similarity = 0.5 * list_similarity + 0.5 * whole_similarity

    # 是否通過檢查 (根據第二個閾值)
    final_pass = math.isclose(final_similarity, thresholds[1], rel_tol=1e-9) or final_similarity >= thresholds[1]
    return final_similarity, final_pass


# Function to compare JSON attributes
def compare_json(expected: dict, actual: dict, thresholds: list):
    """
    回傳一個 dict, 其結構例如:
    {
       "order_number": {
          "similarity": 0.85,
          "threshold": 0.95,
          "pass": False,
          "missing": False,
          "extra": False
       },
       "order_information": {
          "similarity": 0.40,
          "threshold": 0.90,
          "pass": False,
          "missing": False,
          "extra": False
       },
       "additional_messege": {
          "similarity": 0.20,
          "threshold": 0.80,
          "pass": False,
          "missing": False,
          "extra": False
       },
       "some_extra_key_in_output": {
          "similarity": None,
          "pass": False,
          "missing": False,
          "extra": True
       }
    }
    """
    if len(thresholds) != len(expected):
        raise ValueError("Length of thresholds must match the number of keys in expected output.")

    results = {}
    expected_keys = list(expected.keys())

    for idx, key in enumerate(expected_keys):
        if key not in actual:
            # 如果 output 中沒有此 key => 直接視為 missing
            results[key] = {
                "similarity": 0.0,
                "threshold": thresholds[idx],
                "pass": False,
                "missing": True,
                "extra": False
            }
        else:
            # 若是 list vs. list
            if isinstance(expected[key], list):
                similarity, pass_check = hybrid_similarity(
                    expected[key],
                    actual[key],
                    [thresholds[idx], thresholds[idx]]
                )
                results[key] = {
                    "similarity": similarity,
                    "threshold": thresholds[idx],
                    "pass": pass_check,
                    "missing": False,
                    "extra": False
                }
            # 若是單一字串
            elif isinstance(expected[key], str):
                similarity = calculate_similarity(expected[key], actual[key])
                pass_check = math.isclose(similarity, thresholds[idx], rel_tol=1e-9) or similarity >= thresholds[idx]
                results[key] = {
                    "similarity": similarity,
                    "threshold": thresholds[idx],
                    "pass": pass_check,
                    "missing": False,
                    "extra": False
                }

    # 找出多餘的 keys
    extra_keys = set(actual.keys()) - set(expected.keys())
    for key in extra_keys:
        results[key] = {
            "similarity": None,
            "threshold": None,
            "pass": False,
            "missing": False,
            "extra": True
        }

    return results

def display_result(results):
    """顯示 compare_json 的基礎結果(相似度、閾值、通過與否、是否缺失或多餘)。"""
    for key, result in results.items():
        if result.get("missing"):
            print(f"{key}: Missing in output, Pass = False")
        elif result.get("extra"):
            print(f"{key}: Extra key in output, Pass = False")
        else:
            sim = result["similarity"]
            thr = result["threshold"]
            p = result["pass"]
            print(f"{key}: Similarity = {sim:.2f}, Threshold = {thr}, Pass = {p}")

def compute_final_score(expected: dict, actual: dict, results: dict) -> float:
    """
    根據您定義的:
    (E - O)_θ 以及 (O - E)_θ 的大小, 
    計算
    s = 1 - ( (|E-O| + |O-E|) / (|E| + |O|) )

    這裡:
      |E| = expected 的 key 數量
      |O| = actual (output) 的 key 數量
    """
    # 1) 先計算 |E| 與 |O|
    size_E = len(expected)  # 鍵的數量
    size_O = len(actual)

    # 2) 計算 (E-O)_θ 及 (O-E)_θ 的元素個數
    #   - missing_count: 對應 (E - O)_θ
    #   - extra_count:   對應 (O - E)_θ
    missing_count = 0
    extra_count = 0

    for key, info in results.items():
        # 如果這個 key 是 expected 裡的，但 pass=False 就算 missing
        # 或是標記了 "missing": True, 也算 missing
        if key in expected:
            if not info.get("pass"):
                missing_count += 1
                extra_count += 1

        # 如果這個 key 是 output 裡的，但 pass=False 就算 extra
        # 或是標記了 "extra": True, 也算 extra
        if key in actual:
            if info.get("extra"):
                extra_count += 1

    # 3) 套用評分公式 s
    denominator = size_E + size_O
    print(f"Missing count = {missing_count} , Extra count = {extra_count}")
    numerator = (missing_count + extra_count)
    # 避免 denominator = 0 的極端情形
    if denominator == 0:
        return 0.0

    score = 1 - (numerator / denominator)
    return score

if __name__=='__main__':
    # ----------------------- 以下為示範使用 ----------------------------
    expected_output = {
    "order_number": [
        "請輸入您的訂單編號以便查詢。"
    ],
    "order_information": [
        "order_number: 123456, order_amount: NT$ 1,500, shipping_status: 已出貨, shipping_date: 2025-01-18, invoice_link: https://www.invoice-example.com/123456"
    ],
    "additional_messege": [
        "若有進一步的疑問，歡迎再次詢問，我們會盡快為您服務。"
    ]
    }

    output = {
    "order_number": [
        "請輸入您的訂單編號。"
    ],
    "order_information": [
        "order_id: 123456, order_amount: 45.76 usd, shipping_status: 已出貨, shipping_date: 01-18-2025, invoice_link: https://www.invoice-example.com/123456"
    ],
    "additional_messege": [
        "若您需要進一步協助，請撥打客服專線 0800-123-456。"
    ],
    "order_cancellation": [
    "content: 已經幫您取消訂單。"  ]

    }

    # 你可以自行調整 thresholds，這裡依示例給定
    thresholds = [0.95, 0.9, 0.8]

    # 1) 進行 JSON 比較
    results = compare_json(expected_output, output, thresholds)

    # 2) 顯示比對結果
    display_result(results)

    # 3) 根據 (E-O)_θ, (O-E)_θ 計算最終分數
    final_score = compute_final_score(expected_output, output, results)

    print("-------------------------------------------------------")
    print(f"Final Score = {final_score:.3f}")