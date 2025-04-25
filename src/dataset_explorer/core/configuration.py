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

"""Data analysis automation configuration."""

from typing import Dict, List, Optional

from pydantic_settings import BaseSettings


class DataAnalysisSettings(BaseSettings):
    """Base data analysis settings object."""

    supported_file_types_to_loader_map: Dict[str, str] = {
        ".csv": "pandas.read_csv()",
        ".xlsx": "pandas.read_excel()",
        ".json": "dataset_explorer.core.utils.read_json_content()",
    }

    metadata_fields: List[str] = ["description", "citation", "homepage", "license"]
    hf_datasetdict_field_to_analyse: str = "train"
    truncate_metadata_string_character_length: int = 1000

    top_tail_words_count: int = 50
    valid_links_fraction: float = 0.3

    is_pil_image_fraction: float = 0.5
    is_image_fraction: float = 0.5

    multiprocessing: bool = True
    num_processes: int = 6


class ReporterSettings(BaseSettings):
    """Base reporter settings."""

    max_features_to_stats_table: int = 100


class GenAISettings(BaseSettings):
    """Base GenAI settings."""

    language_model_name: str = "llama3"
    language_model_url_endpoint: str = ""
    quantize: bool = False
    provider: str = "ollama"

    text2vec_model_name: str = "sentence-transformers/all-mpnet-base-v2"
    text2vec_max_tokens_per_chunk: int = 512

    image2text_model_name: str = ""

    examples_sample_size: int = 5

    inference_batch_size: int = 32
    examples_file_name: str = "examples.json"

    # Langchain interface settings
    decoding_method: str = "greedy"
    max_new_tokens: int = 2048
    min_new_tokens: int = 20
    temperature: float = 0.5
    stop_sequences: List[str] = ["\Observ", "\Output:", "}"]
    beam_width: Optional[int] = None
    random_seed: int = 42
    repetition_penalty: float = 1.0
    time_limit: int = 10000000
    top_k: Optional[int] = None
    top_p: float = 1
    truncate_input_tokens: int = 0
    typical_p: Optional[int] = None

    system_message_prefix: str = (
        "You are a helpful data analyst. You have been given metadata from a dataset. "
    )

    dataframe_temp: str = ".temp.csv"

    postprocessing_start_token: str = "<START>"
    postprocessing_end_token: str = "<END>"

    attributes: List[str] = [
        "decoding_method",
        "max_new_tokens",
        "min_new_tokens",
        "temperature",
        "stop_sequences",
        "random_seed",
        "top_k",
        "top_p",
        "truncate_input_tokens",
    ]

    metadata_fields_for_prompt: List[str] = [
        "dataset_name",
        "description",
        "paper_content",
        "dataset_examples",
    ]

    max_prompt_length: int = 16000


class ZeonodoCommunitySettings(BaseSettings):
    """Base settings for parsing Zenodo Community Downloads."""

    community_info_json_name: str = "record.json"
    paper_reference_name: str = "publication.pdf"
    zenodo_records_db: str = "community_info_db.json"


NUMERICAL_TYPES = (int, float)

# instantiating the objects
DATA_ANALYSIS_SETTINGS = DataAnalysisSettings()
GEN_AI_SETTINGS = GenAISettings()
ZENODO_SETTINGS = ZeonodoCommunitySettings()
REPORTER_SETTINGS = ReporterSettings()

PROMPTS: Dict[str, str] = {
    "dataset_name": "Propose a one word title for a dataset. Consider the following additional information TO_REPLACE. Answer with a single word, identified by <START> and <END> tokens.",
    "description": "Write a description for the dataset as detailed as possible. When possible, use the information from the paper. Consider the following additional information TO_REPLACE. If possible, say what the data content is. Do not add information about the license.",
    "keywords": "Find synonyms that represent the dataset. Consider this metadata: TO_REPLACE. Return an answer structured as a list of five comma separated words. Add <START> and <END> tokens to identify the list.",
    "domain": "Find a single word that represents the applicative domain of the dataset with these known metadata: TO_REPLACE. Return only a single word, identified by <START> and <END> tokens.",
    "aggregate_captions": "Propose a few sentences summarizing the input types described by the following captions: ADDITIONAL_INFORMATION.",
    "basic": "Propose a string to fill the METADATA_FIELD of the metadata describing the dataset.",
}
