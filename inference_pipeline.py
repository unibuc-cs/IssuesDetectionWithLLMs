# Inference of the model with vLLM. Please setup your own server: https://github.com/vllm-project/vllm
# An example is provided in the main function below.
import requests

VLLM_SERVER = "http://localhost:8000/v1/completions"  # Adjust to your vLLM endpoint
PROMPT_TEMPLATE = """Reddit feedback: {text}
###
"""

def generate_ticket(text):
    payload = {
        "model": "llama-3-8b-instruct",  # Must match what your vLLM server exposes
        "prompt": PROMPT_TEMPLATE.format(text=text),
        "temperature": 0.4,
        "max_tokens": 512,
    }
    response = requests.post(VLLM_SERVER, json=payload)
    return response.json()["choices"][0]["text"]

if __name__ == "__main__":
    test_input = "The game lags when switching weapons and crashes in multiplayer mode."
    output = generate_ticket(test_input)
    print("Predicted output:\n", output)
