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

"""Data utils for automating exploratory dataset analysis"""

import ast
import importlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np  # noqa: F401
import pandas as pd
from datasets import Dataset, DatasetDict, DatasetInfo, load_dataset
from dotenv import load_dotenv

from ..models import create_llm
from .configuration import DATA_ANALYSIS_SETTINGS, GEN_AI_SETTINGS, NUMERICAL_TYPES
from .summarizer import get_summary_map_reduce, precompute_paper_summaries

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(logging.NullHandler())


# make sure you have a .env file under genai root with
# GENAI_KEY=<your-genai-key>
# GENAI_API=<genai-api-endpoint>
load_dotenv()


def read_json_content(path_to_json_file: Path = Path()) -> pd.DataFrame:
    """Json to Pandas Dataframe parser.

    Args:
        path_to_json_file: path to json. Defaults to Path().

    Returns:
        pandas dataframe of the json data.
    """
    with path_to_json_file.open() as file:
        json_content = json.load(file)
        return pd.json_normalize(json_content)


def list_files(data_path: Path = Path()) -> List[Path]:
    """Lists all the files in the data_path folder.

    Args:
        data_path: string of the path. Defaults to "".

    Returns:
        A list of paths.
    """
    sub_files = []

    if not data_path.is_dir():
        return [data_path]

    for folder_path in data_path.iterdir():
        sub_files += list_files(folder_path)

    return sub_files


def keep_only_supported_files(
    files_list: List[Path], supported_types: List[str]
) -> Dict[str, List[Path]]:
    """Removes files that are not in the supported extensions from a list of paths.

    Args:
        files_list: list of paths.
        supported_types: list of supported extensions.

    Returns:
        The filtered list of paths.
    """
    suffix_to_filtered_file: Dict[str, List[Path]] = {}
    unsupported_files = set()
    for file_path in files_list:
        if file_path.suffix in supported_types:
            if file_path.suffix not in suffix_to_filtered_file:
                suffix_to_filtered_file[file_path.suffix] = []
            suffix_to_filtered_file[file_path.suffix].append(file_path)
        else:
            unsupported_files.add(file_path.suffix)
    logger.info(f"Unsupported file extensions: {unsupported_files}")
    return suffix_to_filtered_file


def df_to_hf_dataset(
    pd_dataframe: pd.DataFrame, metadata: Optional[Dict[str, str]] = {}
) -> DatasetDict:
    """Takes a pandas dataframe and a dictionary of metadata and creates a HF dataset.

    Args:
        pd_dataframe: pandas dataframe.
        metadata: dataset metadata. Defaults to "".

    Returns:
        HF dataset dict containing the data in the pandas df.
    """

    hf_dataset = Dataset.from_pandas(pd_dataframe.astype(str))
    dataset_dict = DatasetDict({DATA_ANALYSIS_SETTINGS.hf_datasetdict_field_to_analyse: hf_dataset})
    if metadata:
        for field_name in DATA_ANALYSIS_SETTINGS.metadata_fields:
            if field_name in metadata:
                setattr(
                    dataset_dict[DATA_ANALYSIS_SETTINGS.hf_datasetdict_field_to_analyse].info,
                    field_name,
                    metadata[field_name],
                )
    return dataset_dict


def file_to_hf_dataset(path_to_file: Path) -> DatasetDict:
    """Load data in path_to_file.

    Args:
        path_to_file: path to the file to load.

    Returns:
        HF dataset dict containing the data in the file.
    """

    return load_dataset(path_to_file.suffix.strip("."), data_files=str(path_to_file))


def get_dataset_feature_names(dataset: DatasetDict) -> str:
    """Returns a string of all the feature names of the dataset.

    Args:
        dataset: a HF dataset object.

    Returns:
        the feature names formatted as a string for interence to a language model.
    """
    features_list = map(
        str,
        [
            feature_name
            for feature_name in dataset[
                DATA_ANALYSIS_SETTINGS.hf_datasetdict_field_to_analyse
            ].features.keys()
            if not feature_name.startswith("__")
        ],
    )
    delimiter = ", "
    string_of_features_list = delimiter.join(features_list)
    return string_of_features_list


def get_dataset_label_names(dataset: DatasetDict) -> str:
    """Gets a string of labels of the datast.

    Args:
        dataset: HF dataset.

    Returns:
        list of dataset labels as a string.
    """
    if "label" in dataset[DATA_ANALYSIS_SETTINGS.hf_datasetdict_field_to_analyse].features.keys():
        labels_list = (
            dataset[DATA_ANALYSIS_SETTINGS.hf_datasetdict_field_to_analyse].features["label"].names
        )
        delimiter = ", "
        string_of_labels = delimiter.join(labels_list)
        return string_of_labels.strip()
    return ""


def parse_metadata(hf_metadata: DatasetInfo) -> Dict[str, str]:
    """Parses HF metadata into a dictionary of strings.

    Args:
        hf_metadata: HF dataset.

    Returns:
        Parsed metadata.
    """
    dataset_metadata_to_return = {}
    metadata_fields = [
        field
        for field in dir(hf_metadata)
        if not field.startswith("__") and not callable(getattr(hf_metadata, field))
    ]
    for field in metadata_fields:
        if type(hf_metadata.__getattribute__(str(field))) is str:
            dataset_metadata_to_return[str(field)] = hf_metadata.__getattribute__(str(field))
    return dataset_metadata_to_return


def clean_and_format_string(input_string: str) -> str:
    """Cleans a string from single spaces and special characters.

    Args:
        input_string: _description_.

    Returns:
        _description_.
    """
    cleaned_string = input_string.strip()
    cleaned_string = re.sub(r"\s+", " ", cleaned_string)
    cleaned_string = re.sub(r"[^\w\s]", "", cleaned_string)

    return cleaned_string


def isolate_text_pattern(input_string, pattern) -> Union[str, list[str]]:
    """Uses regular expressions to isolate a pattern.

    Args:
        input_string: input string.
        pattern: string describing the pattern.

    Returns:
        the pattern matches.
    """
    matches = re.findall(pattern, input_string, re.DOTALL)

    return matches if matches else input_string


def get_metadata_string(known_metadata: Dict[str, Any] = {}) -> str:
    """Converts the metadata into a string.

    Args:
        known_metadata: datast metadata. Defaults to {}.

    Returns:
        Converted metadata into a string.
    """
    string = ""

    keys_to_consider = GEN_AI_SETTINGS.metadata_fields_for_prompt
    logger.info(f"Creating prompt metadata string with keys: {keys_to_consider}")
    keys_to_append = []

    for key in keys_to_consider:
        if key not in known_metadata:
            keys_to_consider.remove(key)

    text_features = "text_features_stats"
    if text_features in keys_to_consider:
        # TODO: implement how to add feature stats
        keys_to_consider.remove(text_features)
        logger.error("Text feature stats still not implemented.")

    if "per_image_stats" in keys_to_consider:
        # TODO: implement how to add image stats and captions
        logger.error("Per image stats in prompt non supported.")
        keys_to_consider.remove("per_image_stats")

    if "paper_content" in keys_to_consider:
        keys_to_consider.remove("paper_content")
        keys_to_append.append("paper_content")

    key_delimiter = ": "
    item_delimiter = ", "

    for key in keys_to_consider:
        if key in known_metadata:
            string += key.join(key_delimiter)
            string += str(known_metadata[key])
            string += item_delimiter

    if "paper_content" in keys_to_append:
        output_dir = known_metadata["output_directory"]
        if "zenodo_record_id" in known_metadata:
            paper_file_name = (
                str(output_dir.parent)
                + "/papers/"
                + str(known_metadata["zenodo_record_id"])
                + "-publication-summary.txt"
            )

            if Path(paper_file_name).exists():
                with Path(paper_file_name).open("r") as paper_file:
                    paper_data = paper_file.read()
                    string += "Paper content: "
                    string += paper_data
            elif not Path(
                str(output_dir.parent)
                + "/papers/"
                + str(known_metadata["zenodo_record_id"])
                + "-publication.txt"
            ).exists():
                # there is no paper matching the record
                paper_data = ""
            elif known_metadata["output_directory"].parent.joinpath("papers").exists():
                precompute_paper_summaries(
                    known_metadata["output_directory"].parent.joinpath("papers")
                )  # assumes papers is in the results folder
                if Path(paper_file_name).exists():
                    with Path(paper_file_name).open("r") as paper_file:
                        paper_data = paper_file.read()
                        string += "Paper content: "
                        string += paper_data
                else:
                    paper_data = ""
            else:
                logger.error("Error reading paper. Paper not included.")
                paper_data = ""

    summary = get_summary_map_reduce(string, create_llm())

    return truncate_string(
        summary, DATA_ANALYSIS_SETTINGS.truncate_metadata_string_character_length
    )


def get_latest_edit_date(path_to_folder: Path) -> str:
    """Gets the latest edit date of the files in a folder.

    Args:
        path_to_folder: path to the dataset folder.

    Returns:
        Returns the latest edit date formatted as a string.
    """
    files = list_files(path_to_folder)
    if not files:
        return "Creation date unknown"

    most_recent_modification_time = max([file.stat().st_mtime for file in files])
    return datetime.fromtimestamp(most_recent_modification_time).strftime("%Y-%m-%d %H:%M:%S")


def get_dataset_creation_date_from_folder(path_to_folder: Path) -> str:
    """Gets a path to a dataset folder and finds the latest creation date.

    Args:
        path_to_folder: path to the dataset folder.

    Returns:
        Returns the latest creation date.
    """
    return get_latest_edit_date(path_to_folder=path_to_folder)


def heading(text: str) -> str:
    """Helper function for centering text."""
    return "\n" + f" {text} ".center(80, "=") + "\n"


def get_creation_date(hf_dataset: DatasetDict, known_metadata: Dict[str, Any] = {}):  # noqa: ARG001
    """Given a HF dataset, generates a string with the creation date of the dataset.

    Args:
        hf_dataset: HF dataset dict.
        known_metadata: Dictionary of the known dataset metadata. Defaults to {}.

    Returns:
        Dataset creation date formatted as a string. Returns unknown if it is impossible to recollect the creation date.
    """
    if "creation_date" in known_metadata:
        return known_metadata["creation_date"]
    if "data_folder" in known_metadata:
        known_metadata["creation_date"] = get_dataset_creation_date_from_folder(
            Path(known_metadata["data_folder"])
        )
    else:
        known_metadata["creation_date"] = "Unknown"
    return known_metadata["creation_date"]


def import_and_load_data(package_descriptor: str, data_file_path: Path) -> pd.DataFrame:
    """Import a package.

    Args:
        package_descriptor: name of the package to import and its arguments.

    Returns:
        The package as object and instantiated package object.
    """

    logger.info(f"Processing: {data_file_path}")

    try:
        package_module: str = ".".join(package_descriptor.split(".")[:-1])
        package_name: str = package_descriptor.split(".")[-1].split("(")[0]
        pattern = r"(\w+)\s*=\s*(\w+)"  # pattern to find arguments like '(arg=val)'
        package_arguments: Dict[str, Any] = {
            str(match[0]): eval(match[1]) for match in re.findall(pattern, package_descriptor)
        }
        package_object = getattr(importlib.import_module(package_module), package_name)

    except ImportError as import_error_exception:
        logger.warning(
            f"missing dependencies present, make sure to install them (details: {import_error_exception})"
        )
        raise ImportError(import_error_exception)

    if package_object:
        try:
            loaded_data = package_object(data_file_path, **package_arguments)
            if not isinstance(loaded_data, pd.DataFrame):
                logger.error(f"Loaded data is not a DataFrame as expected: {data_file_path}.")
        except Exception as _:
            logger.error(f"Loading of file failed: {data_file_path}.")
            return pd.DataFrame()

    return loaded_data


def string_to_dict(dict_string: str) -> Dict:
    """Converts a string into a dictionary.

    Args:
        dict_string: input string.

    Returns:
        Input string converted to dictionary.
    """
    string_to_parse = dict_string.replace("nan", "np.nan")
    string_to_parse = string_to_parse.replace("np.int64", "int").replace("np.float64", "float")
    try:
        return ast.literal_eval(string_to_parse)
    except ValueError:
        return eval(string_to_parse)


def string_to_dataframe(dict_string: str) -> pd.DataFrame:
    """Gets a string representing a dictionary and creates a dataframe from it.

    Args:
        dict_string: input string.

    Returns:
        A dataframe.
    """
    string_to_parse = dict_string.replace("nan", "None")
    string_to_parse = string_to_parse.replace("np.int64", "int").replace("np.float64", "float")
    try:
        return pd.DataFrame(ast.literal_eval(string_to_parse))
    except ValueError:
        return pd.DataFrame(eval(string_to_parse))


def is_numerical_feature(
    data: pd.Series,
    numerical_fraction: float = 0.9,
) -> bool:
    """Check whether a feature is numerical with a given threshold.

    Args:
        data: feature data.
        numerical_fraction: fraction of numerical values to consider a feature numerical. Defaults to 0.9.

    Returns:
        whether a feature is numerical.
    """
    numerical_count: int = 0
    non_numerical_count: int = 0
    for value in data:
        try:
            value = float(value)
        except Exception:
            value = value
        if isinstance(value, NUMERICAL_TYPES):
            numerical_count += 1
        else:
            non_numerical_count += 1
    return (
        (numerical_count / float(numerical_count + non_numerical_count)) >= numerical_fraction
        if non_numerical_count > 0
        else True
    )


def truncate_data(data: pd.Series, max_length: int = 5):
    """Gets a pd series of strings and truncates it to a max length.

    Args:
        data: data string.
        max_length: max length. Defaults to 5.

    Returns:
        truncated string.
    """
    value_str = str(data)
    return value_str[:max_length] if len(value_str) > max_length else value_str


def truncate_string(string: str, max_length: int = 15):
    """Gets an input string in truncates it to max_length.

    Args:
        string: input_string.
        max_length: max length. Defaults to 5.
    """
    return string[:max_length] + "."


def get_dataset_formats(hf_dataset: DatasetDict, known_metadata: Dict[str, Any] = {}) -> str:  # noqa: ARG001
    """Given a HF dataset, generates a string that breaksdown the dataset files formats

    Args:
        hf_dataset: HF dataset dict.
        known_metadata: Dictionary of the known dataset metadata. Defaults to {}.

    Returns:
        Dataset creation date formatted as a string. Returns unknown if it is impossible to recollect the creation date.
    """
    if "data_folder" in known_metadata:
        known_metadata["file_formats"] = get_data_files_breakdown(
            Path(known_metadata["data_folder"])
        )
    else:
        known_metadata["file_formats"] = "Data files breakdown not available."
    return known_metadata["file_formats"]


def get_data_files_breakdown(path_to_folder: Path) -> str:
    """Gets the breakdown of the files in a folder.

    Args:
        path_to_folder: path to the dataset folder.

    Returns:
        Returns the file types breakdown for the files in the given folder.
    """
    files = list_files(path_to_folder)
    if not files:
        return "Creation date unknown"

    data_formats: Dict[str, float] = {}

    for file in files:
        if file.suffix in data_formats:
            data_formats[file.suffix] += 1
        else:
            data_formats[file.suffix] = 1

    for suffix in data_formats:
        data_formats[suffix] /= len(files)

    return str(data_formats)


def string_to_list(input_string: Union[List[str], str]) -> List[str]:
    """Takes and input string of comma separated values and converts it into a list of strings.

    Args:
        input_string: input string to convert.

    Returns:
        List of strings.
    """
    if isinstance(input_string, str):
        return ast.literal_eval(input_string)
    return input_string


def retrace_zenodo_record_parent_folder(path: Path) -> Optional[Path]:
    """Given a path, it retraces back the file tree to find the main zenodo record folder.

    Args:
        path: path to a file in a zenodo dataset.

    Returns:
        the main record folder.
    """
    number_regex = re.compile(r"^\d+$")
    for parent in path.parents:
        if number_regex.match(parent.name):
            return parent
    return None
