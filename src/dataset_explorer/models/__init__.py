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

"""Init chat models."""

from typing import Any, Dict, Union

from langchain.chat_models.base import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_ibm import ChatWatsonx
from loguru import logger

from ..core.configuration import GEN_AI_SETTINGS


def create_llm() -> Union[BaseChatModel, BaseLLM, ChatWatsonx]:
    """
    Create and configure a language model interface based on the provider.

    Args:
    model: The model path.
    provider: The provider of the language model
    (e.g., 'watsonx', or any model supported in langchain: https://api.python.langchain.com/en/latest/chat_models/langchain.chat_models.base.init_chat_model.html).
    model_kwargs: Additional keyword arguments for model configuration.

    Returns:
    configured language model interface.
    """
    provider = GEN_AI_SETTINGS.provider

    if provider.lower() == "watsonx":
        from .watsonx import create_watsonx_llm

        logger.info("Using watsonx API")
        return create_watsonx_llm()
    elif provider.lower() == "huggingface":
        from .huggingface import create_huggingface_llm

        logger.info("Using Huggingface API")
        return create_huggingface_llm()
    elif provider.lower() == "ollama":
        from .ollama import create_ollama_llm

        logger.info("Using OLLama")
        return create_ollama_llm()
    else:
        from .local import LocalChatModel, init_local_chat_model

        logger.info("Using local chat model")
        model_parameters: Dict[str, Any] = {}
        for attr in GEN_AI_SETTINGS.attributes:
            model_parameters[attr] = getattr(GEN_AI_SETTINGS, attr)

        local_model = LocalChatModel(GEN_AI_SETTINGS.language_model_name)
        return init_local_chat_model(model=local_model)
