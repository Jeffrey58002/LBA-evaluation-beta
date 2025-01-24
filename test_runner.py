# test_runner.py

import json
from compare_utils import compare_json, compute_final_score, display_result

def run_tests_from_file(filepath):
    """
    filepath: path to a JSON file that looks like:
    {
      "tests": [
        {
          "test_name": "TestCase1",
          "expected_output": { ... },
          "actual_output": { ... },
          "thresholds": [0.95, 0.9, 0.8, ... ]
        },
        ...
      ]
    }
    """

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    test_cases = data.get("tests_suite_01", [])
    #print(test_cases)
    results_summary = []

    for test_case in test_cases:
        test_name = test_case.get("test_name")
        expected_output = test_case.get("expected_output", {})
        actual_output = test_case.get("actual_output", {})
        thresholds = test_case.get("thresholds", [])

        compare_result = compare_json(expected_output, actual_output, thresholds)
        display_result(compare_result)

        score = compute_final_score(expected_output, actual_output, compare_result)
        results_summary.append({
            "test_name": test_name,
            "score": score
        })

        print(f"Test: {test_name}, Score = {score:.3f}")
        print("=" * 50)

    return results_summary


if __name__ == "__main__":
    # 假設我們有個測試檔 test_suite.json
    summary = run_tests_from_file("./test_suite repository/test_suite.JSON")
    print("All tests completed. Summary:")
    for item in summary:
        print(item)