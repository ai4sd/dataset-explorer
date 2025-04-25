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

"""GenAI postprocessing utils for automating exploratory dataset analysis."""

from ..core.configuration import GEN_AI_SETTINGS
from ..core.utils import clean_and_format_string, isolate_text_pattern


def postprocess_model_response(metadata_field: str, model_response: str) -> str:
    """Postprocesses model response based on the expected field.

    Args:
        metadata_field: name of the metadata field of interest.
        model_response: model response to postprocess.

    Returns:
        model response after postprocessing.
    """
    if metadata_field == "dataset_name":
        return get_name_from_response(model_response)
    if (
        GEN_AI_SETTINGS.postprocessing_start_token in model_response
        or GEN_AI_SETTINGS.postprocessing_end_token in model_response
    ):
        return remove_delimiter_tokens(model_response)
    return model_response


def remove_delimiter_tokens(model_response: str = "") -> str:
    """Runs regex expressions to identify the string between start and end tokens.

    Args:
        model_response: response to parse. Defaults to "".

    Returns:
        response cut within the delimiter special tokens
    """
    pattern_string = f"{GEN_AI_SETTINGS.postprocessing_start_token}(.*?){GEN_AI_SETTINGS.postprocessing_end_token}"
    pattern = r"" + pattern_string
    return isolate_text_pattern(model_response, pattern)[0]


def get_name_from_response(model_reponse: str = ""):
    """Gets a model response for the inference of the dataset name and returns a two-words formatted name.

    Args:
        model_reponse: Model response formatted as a string. Defaults to "".

    Returns:
        Max two words string describing the dataset name.
    """
    if model_reponse == "":
        return "Unnamed dataset"
    words = clean_and_format_string(model_reponse).split(" ")
    words = words[:2]
    return "".join(words)
