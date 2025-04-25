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

"""LLM interface for ibm-watsonx."""

import os
from typing import Dict, Union

from ibm_watsonx_ai import Credentials
from langchain_ibm import ChatWatsonx

from ..core.configuration import GEN_AI_SETTINGS

# Set your .env
# HuggingFace
# WATSONX_API_KEY=your_watsonx_api_key
# WATSONX_URL=your_watsonx_url
# WATSONX_PROJECT_ID=your_watsonx_project_id


def create_watsonx_llm() -> ChatWatsonx:
    """
    Create and configure a language model interface using GenAI.
    Returns:
        Configured language model interface.
    """
    credentials = Credentials(
        url="https://us-south.ml.cloud.ibm.com",
        api_key=os.getenv("WATSONX_APIKEY"),
        verify=False,
    )

    model_parameters: Dict[str, Union[float, int, str]] = {}
    for attr in GEN_AI_SETTINGS.attributes:
        model_parameters[attr] = getattr(GEN_AI_SETTINGS, attr)  # type: ignore

    return ChatWatsonx(
        model_id=GEN_AI_SETTINGS.language_model_name,
        url=credentials.get("url"),
        project_id=os.getenv("WATSONX_PROJECT_ID"),
        params={**model_parameters},
    )
