import json

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


if __name__ == "__main__":
    data = load_json("results/tdc_debug/ace_run_20260308_230901_BBB_Martins_offline/detailed_llm_logs/valid/generator_test_eval_19_20260308_233640_286.json")
    print(data)
    