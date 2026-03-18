import json
import openai

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    data = load_json("results/tdc_eval/ace_run_20260311_184108_DILI_eval_only/detailed_llm_logs/test/generator_test_eval_0_attempt_0_20260311_184127_194.json")
    # print(data)

    # 从错误日志中直接提取用过的 prompt 和 model name
    prompt = data["prompt"]
    model = data["model"] 
    # 连接你本地部署的 generator 节点
    client = openai.OpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8001/v1" # 如果不对请改成你开的端口
    )
    print(f"Sending request to model {model} (Prompt char length: {len(prompt)})...")
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10240
        )
        print("Response object:", response)
        print("\nContent:", response.choices[0].message.content)
    except Exception as e:
        print(f"Failed with exception: {e}")
    