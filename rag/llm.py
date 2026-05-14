import ollama


class OllamaLLM:

    def __init__(self, model="qwen2.5:7b"):
        self.model = model

    def __call__(self, prompt):

        completion = ollama.chat(
            model=self.model,
            messages=[{
                    "role": "user", "content": prompt
            }],
            options={
                "temperature": 0.5, "num_predict": 700
            }
        )

        return completion["message"]["content"]

llm = OllamaLLM()
