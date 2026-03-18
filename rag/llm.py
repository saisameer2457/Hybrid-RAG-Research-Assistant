from huggingface_hub import InferenceClient


class HuggingFaceLLM:

    def __init__(self, model="Qwen/Qwen2.5-7B-Instruct"):
        self.client = InferenceClient(
            api_key="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"
        )
        self.model = model

    def __call__(self, prompt):

        completion = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            max_tokens=700,
            temperature=0.5
        )

        return completion.choices[0].message.content


llm = HuggingFaceLLM()
