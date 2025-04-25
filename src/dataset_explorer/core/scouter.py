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

"""Dataset scouting with automatic exploratory dataset analysis"""

import csv
import json
import multiprocessing
from pathlib import Path
from typing import Any, Dict, List, Tuple

import pandas as pd
import tqdm
from datasets import DatasetDict
from loguru import logger

from ..models import create_llm
from ..models.inference import generate_hierarchical_description, generate_metadata_field
from .analyzer import (
    get_data_examples,
    get_path_to_paper,
    get_pdf_to_txt_paper_path,
    get_zenodo_info,
    get_zenodo_record_id,
)
from .configuration import DATA_ANALYSIS_SETTINGS, ZENODO_SETTINGS
from .utils import (
    df_to_hf_dataset,
    get_creation_date,
    get_dataset_label_names,
    import_and_load_data,
    keep_only_supported_files,
    list_files,
    parse_metadata,
    retrace_zenodo_record_parent_folder,
)


def run_analysis_on_hf_dataset(
    hf_dataset: DatasetDict,
    known_metadata: Dict[str, Any],
    output_directory: Path,
    numerical_fraction: float,
    get_predictability: bool = False,
    get_correlations: bool = False,
    is_confidential: bool = False,
):
    """Run analysis on a hf dataset.

    Args:
        hf_dataset: the hf dataset to analyse
        known_metadata: additional metadata of the hf dataset
        output_directory: output directory
        numerical_fraction: fraction of numerical values to consider a feature numerical. Defaults to 0.9
        get_predictability: if the predictability of each feature based on the others should be evaluated. Defaults to False.
        get_correlations: if the cross-correlation of numerical features should be evaluated. Defaults to false.
        is_confidential: if the real data distribution should be masked for confidentiality. Defaults to false.
    """
    logger.warning(
        f"Running with limited set of features get_predictability: {get_predictability}, get_correlations: {get_correlations}, numerical_fraction: {numerical_fraction} to be ignored."
    )
    dataset_report = {}

    dataset_metadata = hf_dataset[DATA_ANALYSIS_SETTINGS.hf_datasetdict_field_to_analyse].info
    parsed_dataset_metadata = parse_metadata(dataset_metadata)
    label_names = get_dataset_label_names(hf_dataset)
    if label_names:
        known_metadata["label_names"] = label_names
    for key in parsed_dataset_metadata:
        known_metadata[key] = parsed_dataset_metadata[key]
    known_metadata["output_directory"] = output_directory

    if "file_type" not in known_metadata:
        known_metadata["file_type"] = "hf_dataset"
    if "format" not in known_metadata:
        known_metadata["format"] = "hf_dataset"

    # NOTE: right now the generation of a name gets never called and the dataset name is either
    # the real name or the file name. Need to check how to improve, if it is even needed to.
    if "dataset_name" in known_metadata:
        dataset_report["dataset_name"] = known_metadata["dataset_name"]

    try:
        language_model = create_llm()
    except Exception as e:
        raise ValueError(f"Failed to instantiate language model. Exception {e}")

    if "dataset_name" not in known_metadata:
        dataset_report["dataset_name"] = generate_metadata_field(
            metadata_field="dataset_name",
            language_model=language_model,
            known_metadata=known_metadata,
        )

    dataset_report["file_type"] = known_metadata["file_type"]
    dataset_report["file_format"] = known_metadata["format"]
    dataset_report["creation_date"] = get_creation_date(hf_dataset, known_metadata=known_metadata)

    dataset_report["dataset_examples"] = get_data_examples(
        hf_dataset, known_metadata=known_metadata, is_confidential=is_confidential
    )

    dataset_report["zenodo_info"] = get_zenodo_info(known_metadata=known_metadata)

    dataset_report["zenodo_record_id"] = get_zenodo_record_id(known_metadata=known_metadata)

    dataset_report["path_to_paper"] = get_path_to_paper(hf_dataset, known_metadata=known_metadata)

    dataset_report["paper"] = get_pdf_to_txt_paper_path(known_metadata=known_metadata)

    inference_fields_list = ["description", "keywords", "domain"]
    for metadata_field in inference_fields_list:
        dataset_report[metadata_field] = generate_metadata_field(
            metadata_field=metadata_field,
            language_model=language_model,
            known_metadata=known_metadata,
        )

    dataset_report_df = pd.DataFrame.from_records([dataset_report], index=[0])

    output_file = Path.joinpath(output_directory, f"{dataset_report['dataset_name']}-report.pkl")
    dataset_report_df = dataset_report_df.reset_index(drop=True)
    dataset_report_df.to_pickle(output_file)

    return str(output_file)


def run_analysis_on_folder(
    data_directory: Path,
    output_directory: Path,
    numerical_fraction: float = 0.9,
    get_correlations: bool = False,
    get_predictability: bool = False,
    known_metadata: Dict[str, Any] = {},
    is_confidential: bool = False,  # noqa: ARG001 # TODO: integrate this in the functions below
) -> str:
    """Runs the analysis on an input directory.

    Args:
        data_directory: input directory.
        output_directory: output directory.
        numerical_fraction: fraction of numerical values to consider a feature numerical. Defaults to 0.9
        get_predictability: if the predictability of each feature based on the others should be evaluated. Defaults to False.
        get_correlations: if the cross-correlation of numerical features should be evaluated. Defaults to false.
        is_confidential: if the real data distribution should be masked for confidentiality. Defaults to false.

    Returns:
        The path to the report for grounding for Q&A on the dataset by RAG.
    """
    parsed_known_metadata_per_file: Dict[str, Any] = {}
    loaded_dataframe_dict: Dict[str, Any] = {}

    files_list = list_files(data_directory)

    logger.info(f"Found {len(files_list)} files in archive.")

    # TODO: reintegrate in a different way?
    # if len(files_list) < 1:
    #     logger.error("No files in folder")

    suffix_to_filtered_file_list = keep_only_supported_files(
        files_list, list(DATA_ANALYSIS_SETTINGS.supported_file_types_to_loader_map.keys())
    )

    total_file_count = 0

    for file_suffix, filtered_file_list in suffix_to_filtered_file_list.items():
        total_file_count += len(filtered_file_list)

    logger.info(f"Total files to analyze : {total_file_count}")

    for file_suffix, filtered_file_list in suffix_to_filtered_file_list.items():
        loaded_dataframe_dict[file_suffix] = {}
        args = [(file_path, data_directory, output_directory) for file_path in filtered_file_list]
        if DATA_ANALYSIS_SETTINGS.multiprocessing:
            with multiprocessing.Pool(processes=DATA_ANALYSIS_SETTINGS.num_processes) as pool:
                results = pool.starmap(load_dataframe_dictionary, args)
        else:
            results = []
            for arg in args:
                results.append(load_dataframe_dictionary(*arg))

        for result in results:
            loaded_dataframe, file_suffix, file_path_str, metadata_to_add = result
            if not loaded_dataframe.empty:
                loaded_dataframe_dict[file_suffix][file_path_str] = loaded_dataframe
                parsed_known_metadata_per_file[file_path_str] = metadata_to_add
        for file_path_str in parsed_known_metadata_per_file:
            parsed_known_metadata_per_file[file_path_str].update(known_metadata)

    for file_suffix in loaded_dataframe_dict:
        if DATA_ANALYSIS_SETTINGS.multiprocessing:
            with multiprocessing.Pool(processes=DATA_ANALYSIS_SETTINGS.num_processes) as pool:
                args = [
                    (  # type: ignore
                        file_name,
                        loaded_dataframe_dict,
                        output_directory,
                        numerical_fraction,
                        get_predictability,
                        get_correlations,
                        is_confidential,
                        parsed_known_metadata_per_file[file_name],
                    )
                    for file_name in loaded_dataframe_dict[file_suffix]
                ]

                _ = pool.starmap(
                    run_analysis_on_file,
                    tqdm.tqdm(args, total=len(loaded_dataframe_dict[file_suffix])),
                )

        else:
            for file_name in loaded_dataframe_dict[file_suffix]:
                _ = run_analysis_on_file(
                    file_name,
                    loaded_dataframe_dict,
                    output_directory,
                    numerical_fraction,
                    get_predictability,
                    get_correlations,
                    is_confidential,
                    parsed_known_metadata_per_file[file_name],
                )

    output_data_files_directories: List[Path] = [
        x for x in output_directory.iterdir() if x.is_dir()
    ]

    metadata_files_to_merge: List[Path] = []
    for output_data_file_directory in output_data_files_directories:
        report_file_list = [
            x
            for x in output_data_file_directory.iterdir()
            if (not x.is_dir() and "examples.json" not in str(x))
        ]
        if report_file_list:
            for report_file in report_file_list:
                metadata_files_to_merge.append(report_file)
    reports_dataframes: List[pd.DataFrame] = [pd.read_pickle(x) for x in metadata_files_to_merge]
    report_for_grounding: pd.DataFrame = pd.concat(reports_dataframes)
    report_for_grounding_filename: Path = Path.joinpath(
        output_directory,
        f"{str(data_directory).replace('/', '-')}_metadata_summary_without_hierarchy.csv",
    )
    report_for_grounding.to_csv(report_for_grounding_filename, quoting=csv.QUOTE_MINIMAL)

    report_for_grounding_with_hierarchy = get_hierarchical_report(
        report_for_grounding, known_metadata=parsed_known_metadata_per_file
    )
    report_for_grounding_with_hierarchy_filename: Path = Path.joinpath(
        output_directory, f"{str(data_directory).replace('/', '-')}_metadata_summary.csv"
    )

    full_grounding_data_filename: Path = Path.joinpath(
        output_directory, f"{str(data_directory).replace('/', '-')}_metadata_summary.pkl"
    )

    report_for_grounding_with_hierarchy.to_pickle(full_grounding_data_filename)
    report_for_grounding_with_hierarchy.to_csv(
        report_for_grounding_with_hierarchy_filename, quoting=csv.QUOTE_MINIMAL
    )

    return str(report_for_grounding_with_hierarchy_filename)


def get_hierarchical_report(report_summary: pd.DataFrame, known_metadata={}) -> pd.DataFrame:
    """Gets a summary report of the scouted files and generates a hierarchical report of the main dataset folder.

    Args:
        report_summary: input report summary (file-level).

    Raises:
        ValueError: if it cannot instantiate the language model.

    Returns:
        the hierarchical report.
    """

    # TODO: to improve in the next sprint with a summarizer module

    report_summary = report_summary.dropna(axis=1, how="all")

    # if "zenodo_record_id" not in report_summary.columns:
    #    logger.info("Non-zenodo entry report being generated.")
    #    return report_summary

    if len(report_summary) <= 1:
        return report_summary
    if "zenodo_record_id" not in report_summary.columns:
        return report_summary
    if (
        len(report_summary["zenodo_record_id"]) <= 1
        or len(report_summary["zenodo_record_id"].unique()) <= 1
    ):
        return report_summary
    try:
        language_model = create_llm()
    except Exception as e:
        raise ValueError(f"Failed to instantiate language model. Exception {e}")

    records = list(known_metadata.keys())

    zenodo_records = set(report_summary["zenodo_record_id"])
    new_entries = []
    for record in records:
        # TODO: implement as general function
        record_id = record.split("-record")[0].split("-")[-1]
        entry = {}
        print(f"Building hierarchical report for record id {record_id}")

        if record_id in zenodo_records:
            if "zenodo_info" in DATA_ANALYSIS_SETTINGS.metadata_fields:
                zenodo_info = report_summary[report_summary["zenodo_record_id"] == record_id][
                    "zenodo_info"
                ].iloc[
                    0
                ]  # zenodo infos are all the same for all the entries associated to one record id
            else:
                zenodo_info = ""
            entry["path_to_paper"] = report_summary[
                report_summary["zenodo_record_id"] == record_id
            ]["path_to_paper"].iloc[0]
            entry["paper"] = report_summary[report_summary["zenodo_record_id"] == record_id][
                "paper"
            ].iloc[0]
            generated_descriptions = report_summary[
                report_summary["zenodo_record_id"] == record_id
            ]["description"]
            generated_keywords = report_summary[report_summary["zenodo_record_id"] == record_id][
                "keywords"
            ].apply(lambda x: set(x.split(", ")))
            if len(generated_keywords) > 0:
                generated_keywords = set.union(*generated_keywords)
            domain = set(
                report_summary[report_summary["zenodo_record_id"] == record_id]["domain"].to_list()
            )
            file_formats = report_summary[report_summary["zenodo_record_id"] == record_id][
                "file_format"
            ]
            file_formats_count = file_formats.value_counts()

            entry["keywords"] = str(generated_keywords)
            entry["domain"] = str(domain)
            entry["file_formats"] = file_formats_count.to_dict()
        else:
            # this builds an entry for records that had no supported formats (based on the webpage JSON dump)
            # or for files that were larger than 800Mb (as of now)
            logger.info(f"Building record entry based on webpage dump for record id: {record_id}")
            with known_metadata[record]["webpage_dump"].open("r") as json_data:
                zenodo_record_data = json.load(json_data)
            zenodo_info = (
                zenodo_record_data["metadata"]["description"]
                if "description" in zenodo_record_data["metadata"]
                else ""
            )
            generated_descriptions = pd.Series([zenodo_info])
        entry["dataset_name"] = record
        entry["file_type"] = "record"
        entry["zenodo_info"] = zenodo_info
        entry["zenodo_record_id"] = record_id

        entry["description"] = generate_hierarchical_description(
            generated_descriptions, language_model, entry
        )
        new_entries.append(entry)

    hierarchical_data_to_append = pd.DataFrame(new_entries)
    return pd.concat([report_summary, hierarchical_data_to_append])


def load_dataframe_dictionary(
    file_path: Path, data_directory: Path, output_directory: Path
) -> Tuple[pd.DataFrame, str, str, Dict[str, Any]]:
    """Loads the file in a dataframe.

    Args:
        file_path: path to file.
        data_directory: path to data.
        output_directory: output path.

    Returns:
        the file data as a dataframe, the file original suffix and path and the metadata.
    """
    file_suffix = file_path.suffix
    file_path_str = str(file_path.relative_to(data_directory)).replace("/", "-")

    output_directory_dataset = output_directory.joinpath(
        file_path_str.split(".")[0].replace(" ", "")
    )

    if output_directory_dataset.exists():
        logger.info(f"Skipping file {file_path_str} as directory already exists")
        return pd.DataFrame(), file_suffix, file_path_str, {}

    loaded_dataframe = import_and_load_data(
        DATA_ANALYSIS_SETTINGS.supported_file_types_to_loader_map[file_suffix], file_path
    )
    zenodo_record_parent_folder = retrace_zenodo_record_parent_folder(file_path)
    metadata_to_add: Dict[str, Any] = {}
    if zenodo_record_parent_folder:
        if zenodo_record_parent_folder.joinpath(ZENODO_SETTINGS.community_info_json_name).exists():
            metadata_to_add["webpage_dump"] = zenodo_record_parent_folder.joinpath(
                ZENODO_SETTINGS.community_info_json_name
            )

        metadata_to_add["path_to_paper"] = zenodo_record_parent_folder.joinpath(
            ZENODO_SETTINGS.paper_reference_name
        )
        metadata_to_add["zenodo_record_id"] = str(zenodo_record_parent_folder.name)
        metadata_to_add["zenodo_folder"] = zenodo_record_parent_folder

    loaded_dataframe.columns = loaded_dataframe.columns.str.replace("^_+", "", regex=True)
    return loaded_dataframe, file_suffix, file_path_str, metadata_to_add


def run_analysis_on_file(
    file_name: str,
    loaded_dataframe_dict: Dict[str, Any],
    output_directory: Path,
    numerical_fraction: float = 0.9,
    get_predictability: bool = False,
    get_correlations: bool = False,
    is_confidential: bool = False,
    known_metadata={},
):
    """runs the analysis on the file and generates the outputs in output directory.

    Args:
        file_name: string of the path to file.
        loaded_dataframe_dict: loaded dataframe dictionary.
        output_directory: output directory.
        numerical_fraction: the numerical fraction used in the statistical analyses. Defaults to 0.9.
        get_correlations: if the analysis should include correlations. Defaults to False.
        get_predictability: if the analysis should include feature predictability. Defaults to False.
        known_metadata: dictionary of known metadata. Defaults to {}.
    """

    def store_zenodo_info(
        file_name: str, output_directory: Path, known_metadata: Dict[str, Any] = {}
    ) -> None:
        if "zenodo_info" not in DATA_ANALYSIS_SETTINGS.metadata_fields:
            return
        if Path(output_directory.joinpath(ZENODO_SETTINGS.zenodo_records_db)).exists():
            try:
                with Path(output_directory.joinpath(ZENODO_SETTINGS.zenodo_records_db)).open(
                    "r"
                ) as records_db:
                    records_data = json.load(records_db)
            except Exception as e:
                logger.error(
                    f"Could not load zenodo DB in {Path(output_directory.joinpath(ZENODO_SETTINGS.zenodo_records_db))}. Exception {e}. Starting from blank."
                )
                records_data = {}
        else:
            records_data = {}
        data: Dict[str, Any] = {}
        data[file_name] = {}
        if "webpage_dump" in known_metadata:
            with known_metadata["webpage_dump"].open() as json_metadata:
                data[file_name]["zenodo_info"] = json.load(json_metadata)
            data[file_name]["path_to_paper"] = str(known_metadata["path_to_paper"])
            data[file_name]["zenodo_record_id"] = str(known_metadata["zenodo_record_id"])
            data[file_name]["zenodo_folder"] = str(known_metadata["zenodo_folder"])
        records_data.update(data)
        with Path(output_directory.joinpath(ZENODO_SETTINGS.zenodo_records_db)).open(
            "w"
        ) as records_db:
            json.dump(records_data, records_db)
        return

    if ZENODO_SETTINGS.community_info_json_name in file_name:
        store_zenodo_info(file_name, output_directory, known_metadata=known_metadata)
        return

    known_metadata["dataset_name"] = file_name
    known_metadata["file_type"] = "file"
    file_suffix = Path(file_name).suffix
    known_metadata["format"] = file_suffix
    output_directory_dataset = output_directory.joinpath(file_name.split(".")[0].replace(" ", ""))

    loaded_dataframe = loaded_dataframe_dict[file_suffix][file_name]
    hf_dataset = df_to_hf_dataset(loaded_dataframe, metadata=known_metadata)

    output_directory_dataset.mkdir(parents=True, exist_ok=True)

    run_analysis_on_hf_dataset(
        hf_dataset,
        known_metadata,
        output_directory_dataset,
        numerical_fraction,
        get_correlations=get_correlations,
        get_predictability=get_predictability,
        is_confidential=is_confidential,
    )
    logger.info(f"Analysis done: file {file_name}.")
    known_metadata = {}
    return
