# compare_utils.py

import math
import numpy as np
from itertools import product
from embedding_utils import calculate_similarity, compare_as_whole_text

def hybrid_similarity(expected_list, actual_list, thresholds):
    whole_similarity, whole_pass = compare_as_whole_text(
        expected_list,
        actual_list,
        thresholds[1]
    )

    final_similarity = whole_similarity
    final_pass = math.isclose(final_similarity, thresholds[1], rel_tol=1e-9) or final_similarity >= thresholds[1]
    return final_similarity, final_pass


def compare_json(expected: dict, actual: dict, thresholds: list):
    if len(thresholds) != len(expected):
        raise ValueError("Length of thresholds must match the number of keys in expected output.")

    results = {}
    expected_keys = list(expected.keys())

    for idx, key in enumerate(expected_keys):
        if key not in actual:
            results[key] = {
                "similarity": 0.0,
                "threshold": thresholds[idx],
                "pass": False,
                "missing": True,
                "extra": False
            }
        else:
            # If the test case is a list
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
            # If the test case is string
            elif isinstance(expected[key], str):
                similarity = calculate_similarity(expected[key], actual[key])
                pass_check = math.isclose(similarity, thresholds[idx], rel_tol=1e-9) or (similarity >= thresholds[idx])
                results[key] = {
                    "similarity": similarity,
                    "threshold": thresholds[idx],
                    "pass": pass_check,
                    "missing": False,
                    "extra": False
                }

    # Find the extra output results
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
    s = 1 - ((|E-O| + |O-E|) / (|E| + |O|))
    """
    size_E = len(expected)
    size_O = len(actual)

    missing_count = 0
    extra_count = 0

    for key, info in results.items():
        # 如果這個 key 是 expected 裡的，但 pass=False 就算 missing
        if key in expected:
            if not info.get("pass"):
                # 如何計算 missing vs extra
                missing_count += 1
                extra_count += 1

        # 如果這個 key 是 output 裡的，多餘key算 extra
        if key in actual:
            if info.get("extra"):
                extra_count += 1

    denominator = size_E + size_O
    if denominator == 0:
        return 0.0

    score = 1 - ((missing_count + extra_count) / denominator)
    return score
