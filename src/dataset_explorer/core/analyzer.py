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

"""Data analysis basic functionalities for features that were not identified as a specific type for automating exploratory dataset analysis"""

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd
import requests
from datasets import DatasetDict
from docling.document_converter import DocumentConverter
from loguru import logger
from pydantic import Field, create_model

from ..core.configuration import DATA_ANALYSIS_SETTINGS, GEN_AI_SETTINGS, ZENODO_SETTINGS


def get_sample(hf_dataset: DatasetDict, index) -> Dict[str, Any]:
    """Give a HF dataset and an index, returns the corresponding sample.

    Args:
        hf_dataset: HF dataset.
        index: index of interest.

    Returns:
        sample at the specified index.
    """
    data_samples = hf_dataset["train"]
    return data_samples[index]


def filter_examples(
    hf_dataset: DatasetDict,
    feature_names: List[str],
    known_metadata: Dict[str, Any] = {},
    metadata_replacement_field="per_image_stats",
) -> List[Dict[str, Any]]:
    """Filters the features to mask for privacy or replacement of modalities with a given metadata replacement field .

    Args:
        hf_dataset: HF dataset.
        feature_names: names of the features to filter.
        known_metadata: known metadata to use for the replacement in the filter. Defaults to {}.
        metadata_replacement_field: field of the metadata to use for the replacement. Defaults to "per_image_stats".

    Returns:
        _description_.
    """
    replacements = known_metadata[metadata_replacement_field].dropna()
    indexes = replacements.index
    samples = [get_sample(hf_dataset, index) for index in indexes]
    for iterator, index in enumerate(indexes):
        for feature_to_describe in feature_names:
            samples[iterator][feature_to_describe] = str(replacements.loc[index])
    return samples


def get_data_examples(
    hf_dataset: DatasetDict, known_metadata: Dict[str, Any] = {}, is_confidential: bool = False
) -> List[Dict[str, Any]]:
    """Gets data examples from the dataset and filters out / masks the fields that are confidential or to convert to text.

    Args:
        hf_dataset: hugging face dataset.
        known_metadata: known metadata dictionary. Defaults to {}.
        is_confidential: boolean to handle data masking for privacy. Defaults to False.

    Returns:
        a list of data examples as a JSON dump.
    """
    if is_confidential:
        return []

    if len(hf_dataset["train"]) > GEN_AI_SETTINGS.examples_sample_size:
        samples_generator = (
            hf_dataset["train"]
            .shuffle(seed=GEN_AI_SETTINGS.random_seed)
            .select(range(GEN_AI_SETTINGS.examples_sample_size))
        )
        filtered_samples = []
        for index in range(GEN_AI_SETTINGS.examples_sample_size):
            sample = {}
            for feature in samples_generator[:].keys():
                sample[feature] = samples_generator[feature][index]
            filtered_samples.append(sample)
        examples_file = Path(known_metadata["output_directory"]).joinpath(
            f"{known_metadata['dataset_name']}-{GEN_AI_SETTINGS.examples_file_name}"
        )
        logger.info(f"Saving examples in file: {examples_file}")
        with examples_file.open(mode="w") as json_file:
            json.dump(filtered_samples, json_file)

        known_metadata["dataset_examples"] = filtered_samples

        return filtered_samples

    return []


def get_dataset_schema(known_metadata: Dict[str, Any] = {}) -> dict[str, Any]:
    """Determines a pydantic base mode form the dataset and generates a schema.

    Args:
        known_metadata: known metadata dictionary. Defaults to {}.

    Returns:
        A dict with the base model schema.
    """

    # Get examples
    if "dataset_examples" not in known_metadata:
        logger.warning(
            "known_metadata has no key 'dataset_examples', the returned schema for the dataset pydantic model will be empty."
        )
        return {}

    examples = known_metadata["dataset_examples"]

    if len(examples) == 0:
        raise NotImplementedError(
            "Saving a json schema for the pydantic dataset base model is only implemented if there are examples but received {n_examples} examples."
        )

    # Get pydantic dynamic fields from known_metadata
    dynamic_fields: dict = {}
    # NOTE: we are assuming the first example will have all the columms, it is fine if filled with NaN, we just want the column
    for column in examples[0]:
        # Numerical
        if column in known_metadata["features_numeric"]:
            type_column = float

        if column in known_metadata["features_non_numeric"]:
            type_column = str  # type: ignore[assignment]

        if column in known_metadata["text_features"]:
            type_column = str  # type: ignore[assignment]

        # Categorical
        if column in known_metadata["features_categorical"]:
            unique_values: List[str] = list(
                known_metadata["non_numerical_stats_info"]["non_numerical_stats_counts"][
                    column
                ].keys()
            )
            type_column = Literal[tuple(unique_values)]  # type: ignore[assignment] # tuple needed otherwise list unhashable type

        # TODO: image_features

        # Assign type to field
        dynamic_fields[column] = (
            type_column,
            Field(description=f"Dataset feature named {column}."),
        )  # TODO: get column description, column name is placeholder

    # Put a default dataset called "DEFAULT DATASET NAME"
    dataset_name = known_metadata.get("dataset_name", "DEFAULT DATASET NAME")

    # Create dynamic pydantic base model
    pydantic_dataset_base_model = create_model(
        dataset_name,
        **dynamic_fields,
    )

    json_schema = pydantic_dataset_base_model.model_json_schema()

    return json_schema


def parse_link(value: str) -> Union[str, List[str]]:
    """Parses a value of a cell in a pandas dataframe that contains at least one link or path to a file.

    Args:
        value: value of a cell in a pandas dataframe.

    Returns:
        The list of URLs or Paths that are rejoinable.
    """
    link_pattern = r"(https?://[^\s]+?\.(?:pdf|docx|xlsx|png|jpg|jpeg|gif|zip|rar|txt|csv|mp3|mp4))"
    file_links = re.findall(link_pattern, value)
    return file_links


def is_valid_link(value: Any) -> bool:
    """Determines if the value of a cell in a pands dataframe is a link to a valid path.

    Args:
        value: The value from a pandas dataframe cell.

    Returns:
        True if the path exists or the URL can be joined.
    """
    if type(value) is not str:
        return False

    parsed_values = parse_link(value)
    parsed_values_responses = {}

    for parsed_value in parsed_values:
        if any(parsed_value.startswith("http") for parsed_value in parsed_values):
            try:
                response = requests.head(parsed_value, allow_redirects=True, timeout=5)
                parsed_values_responses[parsed_value] = response.status_code == 200
            except requests.RequestException:
                parsed_values_responses[parsed_value] = False

        elif any(
            suffix in parsed_value
            for suffix in DATA_ANALYSIS_SETTINGS.supported_file_types_to_loader_map
        ):
            parsed_values_responses[parsed_value] = Path(parsed_value).exists()
        else:
            parsed_values_responses[parsed_value] = False

    return any(parsed_values_responses.values())


def get_links_to_files_feature_names(
    hf_dataset: DatasetDict, known_metadata: Dict[str, Any] = {}
) -> str:
    """Gets names of the features that contain links to other joinable data files.

    Args:
        hf_dataset: HF dataset.
        known_metadata: Metadata that are already known. Defaults to {}.

    Returns:
        A string containing a list of the feature names that contain links to joinable data files.
    """

    data_frame = pd.DataFrame(hf_dataset[DATA_ANALYSIS_SETTINGS.hf_datasetdict_field_to_analyse])

    if "features_non_numeric" in known_metadata:
        features_to_analyze = set(known_metadata["features_non_numeric"]).intersection(
            data_frame.columns
        )
    else:
        features_to_analyze = set(data_frame.columns)

    links_to_files_features_list = []
    for column in features_to_analyze:
        if (
            sum(data_frame[column].apply(is_valid_link)) / len(data_frame[column])
            > DATA_ANALYSIS_SETTINGS.valid_links_fraction
        ):
            links_to_files_features_list.append(column)

    logger.info(f"Features with links to other files detected: {links_to_files_features_list}")
    known_metadata["links_to_files_features_list"] = links_to_files_features_list
    return str(known_metadata["links_to_files_features_list"])


def get_pdf_to_txt_paper_path(
    path_to_file: Optional[Path] = None, known_metadata: Dict[str, Any] = {}
) -> str:
    """Gets the content of a pdf file with Docling.

    Args:
        path_to_file: Path to a pdf file. Defaults to None.
        known_metadata: known metadata dictionary. Defaults to {}.

    Returns:
        The parsed pdf content as a string or an empty string if the file was not found.
    """

    if "output_directory" not in known_metadata:
        raise ValueError("Missing output directory where to store the results of parsing the PDFs.")
    if "dataset_name" not in known_metadata:
        raise ValueError("Unspecified dataset name.")
    known_metadata["paper"] = ""

    if path_to_file:
        known_metadata["path_to_paper"] = str(path_to_file)

    if "zenodo_record_id" in known_metadata:
        txt_paper_path = Path(
            known_metadata["output_directory"]
            .parent.joinpath("papers")
            .joinpath(
                known_metadata["zenodo_record_id"] + "-" + ZENODO_SETTINGS.paper_reference_name
            )
        ).with_suffix(".txt")
    else:
        txt_paper_path = Path(ZENODO_SETTINGS.paper_reference_name).with_suffix(".txt")

    if txt_paper_path.exists():
        with txt_paper_path.open("r") as paper_txt:
            known_metadata["paper_content"] = paper_txt.read()
            known_metadata["path_to_paper"] = txt_paper_path
    elif "path_to_paper" in known_metadata:
        if Path(known_metadata["path_to_paper"]).exists():
            converter = DocumentConverter()
            result = converter.convert_single(Path(known_metadata["path_to_paper"]))
            pdf_content = result.render_as_markdown()
            with txt_paper_path.open(mode="w", encoding="utf-8") as file:
                file.write(pdf_content)
            known_metadata["path_to_paper"] = txt_paper_path
            known_metadata["paper_content"] = pdf_content

    return known_metadata["path_to_paper"] if "path_to_paper" in known_metadata else ""


def get_pdf_paper_content(paper_path: Optional[Path] = None, known_metadata={}):
    """Returns paper content from metadata.

    Args:
        known_metadata: metadata dictionary. Defaults to {}.

    Returns:
        paper content as a string.
    """

    if paper_path and paper_path.exists():
        _ = get_pdf_to_txt_paper_path(path_to_file=paper_path, known_metadata=known_metadata)
    return known_metadata["paper_content"] if "paper_content" in known_metadata else {}


def get_path_to_paper(
    hf_dataset: DatasetDict,  # noqa: ARG001
    known_metadata: Dict[str, Any] = {},
) -> Optional[Path]:
    """Gets a HF dataset with known metadata and returns the path to the reference paper, if this exists.

    Args:
        hf_dataset: HF data.
        known_metadata: known metadata dictionary. Defaults to {}.

    Returns:
        path to paper if it can be found.
    """

    # TODO: add support to search for the arxiv citation in HF dataset description

    if "path_to_paper" not in known_metadata:
        return None

    return known_metadata["path_to_paper"] if known_metadata["path_to_paper"].exists() else None


def get_zenodo_info(known_metadata: Dict[str, Any] = {}) -> str:
    """Get the information from the zenodo page, if they exists.

    Args:
        known_metadata: known metadata dictionary. Defaults to {}.

    Returns:
        Zenodo page description of the dataset.
    """
    if "zenodo_info" in known_metadata:
        return known_metadata.get("zenodo_info", {}).get("metadata", {}).get("description")
    if "webpage_dump" in known_metadata:
        with known_metadata["webpage_dump"].open("r") as webpage_json:
            zenodo_record_data = json.load(webpage_json)
            known_metadata["zenodo_info"] = (
                zenodo_record_data["metadata"]["description"]
                if "description" in zenodo_record_data["metadata"]
                else ""
            )
            return known_metadata["zenodo_info"]
    return "No info available."


def get_zenodo_record_id(known_metadata: Dict[str, Any] = {}) -> str:
    """Get zenodo record ID, where applicable.

    Args:
        known_metadata: known metadata dictionary. Defaults to {}.

    Returns:
        the zenodo record id where applicable or the empty string.
    """
    if "zenodo_record_id" in known_metadata:
        return known_metadata["zenodo_record_id"]
    elif "webpage_dump" in known_metadata:
        with known_metadata["webpage_dump"].open("r") as webpage_json:
            zenodo_record_data = json.load(webpage_json)
            known_metadata["zenodo_record_id"] = zenodo_record_data["id"]
            return known_metadata["zenodo_record_id"]
    return ""
