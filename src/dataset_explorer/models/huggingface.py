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

"""LLM interface for huggingface."""

import os
from typing import Dict, Union

from huggingface_hub import login
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

from ..core.configuration import GEN_AI_SETTINGS

# set your .env file to have
# HUGGINGFACEHUB_API_TOKEN=your_huggingface_token


def create_huggingface_llm() -> ChatHuggingFace:
    """
    Create and configure a chat language model interface using HuggingFaceEndpoint.

    Args:
        model: The model ID to be used (e.g., "HuggingFaceH4/zephyr-7b-beta").

    Returns:
        Configured language model interface.
    """
    token = os.getenv("HUGGINGFACEHUB_API_TOKEN")

    model_parameters: Dict[str, Union[float, int, str]] = {}
    for attr in GEN_AI_SETTINGS.attributes:
        model_parameters[attr] = getattr(GEN_AI_SETTINGS, attr)  # type: ignore

    llm = HuggingFaceEndpoint(
        repo_id=GEN_AI_SETTINGS.language_model_name,
        huggingfacehub_api_token=login(token),
        **model_parameters,
    )
    return ChatHuggingFace(llm=llm, verbose=True)
