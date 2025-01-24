# main.py
from compare_utils import compare_json, compute_final_score, display_result

if __name__ == "__main__":
    # 這裡舉例我們用硬编码的 expected_output 與 output 來測試
    expected_output = {
        "order_number": [
            "請輸入您的訂單編號以便查詢。"
        ],
        "order_information": [
            "order_number: 123456, order_amount: NT$ 1,500..."
        ],
        "additional_messege": [
            "若有進一步的疑問..."
        ]
    }

    output = {
        "order_number": [
            "請輸入您的訂單編號讓我查詢。"
        ],
        "order_information": [
            "order_id: 123456, order_amount: 45.76 usd..."
        ],
        "additional_messege": [
            "若您需要進一步協助..."
        ],
        "order_cancellation": [
            "content: 已經幫您取消訂單。"
        ]
    }

    thresholds = [0.95, 0.9, 0.8]

    # 執行 compare
    results = compare_json(expected_output, output, thresholds)
    display_result(results)

    final_score = compute_final_score(expected_output, output, results)
    print("-------------------------------------------------------")
    print(f"Final Score = {final_score:.3f}")