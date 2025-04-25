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
"""Set of functions to perform dataset metadata generation augmented with language models."""

from typing import Any, Dict, Optional, Union

import pandas as pd
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from loguru import logger

from ..core.configuration import GEN_AI_SETTINGS, PROMPTS
from ..core.utils import (
    get_metadata_string,
)
from .processing import postprocess_model_response


def generate_metadata_field(
    metadata_field: str,
    language_model: Optional[Union[BaseChatModel, BaseLLM]],
    known_metadata: Dict[str, Any] = {},
) -> str:
    """Generates a metadata field with a langchain interfaced language model.

    Args:
        metadata_field: the identifier of the metadata field that should be filled in through inference.
        language_model: the configured language model interface to use for inference.
        known_metadata: dictionary of known dataset information. Defaults to {}.

    Returns:
        A string that can be used to fill the metadata field of interest.
    """
    if not language_model:
        logger.error("No model found")
        return ""

    response = language_model.invoke(
        get_prompt_message(metadata_field, known_metadata=known_metadata)
    )

    if hasattr(response, "content"):
        known_metadata[metadata_field] = postprocess_model_response(
            metadata_field, response.content
        )
    else:
        known_metadata[metadata_field] = postprocess_model_response(metadata_field, response)

    return known_metadata[metadata_field]


def get_prompt_message(
    metadata_field: str, known_metadata: Dict[str, Any] = {}, additional_information: str = ""
) -> str:
    """Generates a prompt to instruct the language model to fill the metadata field.

    Args:
        metadata_field: name of the metadata field to fill.
        known_metadata: dictionary of known dataset information. Defaults to {}.

    Returns:
        A prompt that can be used to instruct the language model to fill the metadata field of interest.
    """
    prompt = PROMPTS[metadata_field] if metadata_field in PROMPTS else PROMPTS["basic"]

    if "TO_REPLACE" in prompt:
        prompt = prompt.replace("TO_REPLACE", get_metadata_string(known_metadata))
    if "ADDITIONAL_INFORMATION" in prompt:
        logger.info("Replacing ADDITIONAL_INFORMATION")
        prompt = prompt.replace("ADDITIONAL_INFORMATION", additional_information)
    if "METADATA_FIELD" in prompt:
        prompt = prompt.replace("METADATA_FIELD", metadata_field)

    return [("system", GEN_AI_SETTINGS.system_message_prefix), ("human", prompt)]


def generate_aggregated_captions(
    caption_data: pd.Series,
    language_model: Union[BaseChatModel, BaseLLM],
) -> str:
    """Given a pd series of captions, generate an aggregated description.

    Args:

        model_id: model name. Defaults to "".

    Returns:
        A list of keywords describing the dataset.
    """
    data = caption_data.dropna()
    data_str = str([caption for caption in data])
    if not language_model:
        return data_str

    response = language_model.invoke(
        get_prompt_message("aggregate_captions", additional_information=data_str)
    )

    return postprocess_model_response("aggregate_captions", response)


def generate_hierarchical_description(
    descriptions: pd.Series,
    language_model: Optional[Union[BaseChatModel, BaseLLM]],
    known_metadata: Dict[str, Any] = {},
) -> str:
    """Generates a hierarchical description for a set of records with a langchain interfaced language model.

    Args:
        metadata_field: the identifier of the metadata field that should be filled in through inference.
        language_model: the configured language model interface to use for inference.
        known_metadata: dictionary of known dataset information. Defaults to {}.

    Returns:
        A string that can be used to fill the metadata field of interest.
    """
    if not language_model:
        logger.error("No model found")
        return ""

    prompt = "Describe these files. Be as detailed as possible on the file content."
    data = ""
    for key in known_metadata:
        data += str(key)
        data += str(known_metadata[key])
    prompt += data
    prompt += "Here per-file descriptions."
    descriptions = descriptions.tolist()
    index = 0

    while len(prompt) < GEN_AI_SETTINGS.max_prompt_length and len(descriptions) > index:
        prompt += descriptions[index]
        index += 1

    response = language_model.invoke(prompt)
    logger.info(f"Response: {response}")

    if hasattr(response, "content"):
        return response.content
    return response
