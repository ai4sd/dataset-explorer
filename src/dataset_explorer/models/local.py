# MIT License

# Copyright (c) 2024 - IBM Research

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Init of local chat models."""

from typing import Dict, Union

import torch
from langchain.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, pipeline

from ..core.configuration import GEN_AI_SETTINGS


class LocalChatModel:
    """Local chat model class to run without API need."""

    def __init__(self, model_path: str):
        """Initializes a local chat model.

        Args:
            model_path: model path on HF.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        if GEN_AI_SETTINGS.quantize:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

            self.model = AutoModelForCausalLM.from_pretrained(
                model_path, device_map="auto", quantization_config=quant_config
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map={"": "cpu"},
            )
        model_parameters: Dict[str, Union[float, int, str]] = {}
        for attr in GEN_AI_SETTINGS.attributes:
            model_parameters[attr] = getattr(GEN_AI_SETTINGS, attr)  # type: ignore
        self.generation_args = model_parameters
        self.pipeline = pipeline(
            "text-generation", model=self.model, tokenizer=self.tokenizer, return_full_text=True
        )
        self.generation_args = model_parameters

    def create_pipeline(self):
        """create inference pipeline.

        Returns:
            output of generate response.
        """
        return lambda prompt: self.generate_response(prompt)

    def generate_response(self, prompt: str):
        """performs local inference.

        Args:
            prompt: input prompt.

        Returns:
            output string.
        """
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(**inputs, **self.generation_args)
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response


def init_local_chat_model(model: str):
    """Initializes local chat model.

    Args:
        model: pointer to local model name.

    Returns:
        a local model pipeline.
    """
    if isinstance(model, LocalChatModel):
        return HuggingFacePipeline(
            pipeline=model.pipeline,
            model_kwargs=model.generation_args,
        )
    raise ValueError("Pass a local model.")
