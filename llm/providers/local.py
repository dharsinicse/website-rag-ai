from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from llm.base import LLMProvider


class LocalLLMProvider(LLMProvider):
    """Local Hugging Face Seq2Seq LLM provider."""

    def __init__(
        self,
        model_name: str,
        max_input_length: int = 2048,
    ):
        self.model_name = model_name
        self.max_input_length = max_input_length

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            model_name
        )

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
    ) -> str:
        if not prompt.strip():
            raise ValueError("Prompt cannot be empty")

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )

        outputs = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )

        answer = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True,
        )

        return answer.strip()