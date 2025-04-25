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

"""Entry point to cli scouter."""

import ast
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List

import click
from datasets import load_dataset
from loguru import logger

from dataset_explorer.agent.core import create_agent_executor
from dataset_explorer.agent.tools.rag_tools import TOOLS
from dataset_explorer.core.rag import (
    create_vector_store_from_json_file,
    create_vector_store_from_metadata,
)
from dataset_explorer.models import create_llm

from .core.rag import merge_indexes
from .core.scouter import run_analysis_on_folder, run_analysis_on_hf_dataset

warnings.filterwarnings("ignore")


def run_agent_cli():
    """Runs the agent CLI."""

    llm = create_llm()
    agent_executor = create_agent_executor(TOOLS, llm)

    logger.info("Agent started. Type 'exit' to quit.")

    while True:
        input_text = input("Enter your question: ")
        if input_text.lower() == "exit":
            return
        response = agent_executor.invoke({"input": input_text})
        logger.info(f"Agent response: {response.get('output')}")


@click.command()
@click.option(
    "--path_to_file",
    help="path to report file",
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--output_directory",
    help="output directory for the vector store",
    required=False,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--nr_lines",
    help="consider only first nr lines",
    required=False,
    type=int,
)
def run_vector_store_creation(
    path_to_file: Path, output_directory: Path | None = None, nr_lines: int | None = None
):
    """Runs vector store creation"""

    if path_to_file.suffix in [".json", ".jsonl"]:
        function_to_run: callable = create_vector_store_from_json_file
    else:
        function_to_run: callable = create_vector_store_from_metadata
    kwargs = {}
    if len(sys.argv) > 2:
        kwargs["nr_lines"] = nr_lines
    else:
        kwargs["nr_lines"] = None

    function_to_run(path_to_file=path_to_file, output_directory=output_directory, **kwargs)


@click.command()
@click.argument("--paths", nargs=-1, type=click.Path())
@click.option(
    "--output_directory",
    help="output directory for the merged vector store",
    required=False,
    type=click.Path(path_type=Path, exists=True),
)
def merge_vector_stores(__paths: Path, output_directory: Path | None = None):
    """Merges vector stores.

    Args:
        __paths: list of paths to the vector store directories.
        output_directory: _description_. Defaults to None.
    """
    indexes: List[Path] = []
    for path in __paths:
        indexes.append(Path(path))
    merge_indexes(indexes, output_directory=output_directory)
    return


@click.command()
@click.option(
    "--dataset_name",
    help="HF dataset name",
    required=True,
    type=str,
)
@click.option(
    "--output_directory",
    help="output directory for the report",
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--hf_load_dataset_configuration",
    help="HF dataset configuration",
    required=False,
    type=str,
    default="{}",
)
@click.option(
    "--known_metadata",
    help="known metadata of the dataset",
    required=False,
    type=str,
    default="{}",
)
@click.option(
    "--numerical_fraction",
    help="fraction of numerical values to consider a feature numerical",
    default=0.9,
    type=float,
)
@click.option(
    "--get_predictability",
    help="whether to run a target predictability analysis on tabular features",
    default=False,
    type=float,
    is_flag=True,
)
@click.option(
    "--get_correlations",
    help="whether to run a pairwise feature correlation analysis on tabular features",
    default=False,
    type=float,
    is_flag=True,
)
def load_and_run_data_analysis_on_hf_dataset(
    dataset_name: str,
    output_directory: Path,
    hf_load_dataset_configuration: str = "{}",
    known_metadata: str = "{}",
    numerical_fraction: float = 0.9,
    get_predictability: bool = False,
    get_correlations: bool = False,
) -> None:
    """Runs a basic data analysis script that generates a dataset report with metadata.

    Args:
        dataset_name: HF dataset name.
        output_directory: output directory for the report.
        hf_load_dataset_configuration: HF dataset configuration.
        known_metadata: known metadata of the dataset.
        numerical_fraction: fraction of numerical values to consider a feature numerical. Defaults to 0.9.
    """
    parsed_hf_load_dataset_configuration: Dict[str, Any] = ast.literal_eval(
        hf_load_dataset_configuration
    )
    parsed_known_metadata: Dict[str, Any] = ast.literal_eval(known_metadata)

    hf_dataset = load_dataset(dataset_name, **parsed_hf_load_dataset_configuration)

    logger.info(f"dataset {dataset_name} has been loaded.")

    parsed_known_metadata["dataset_name"] = dataset_name

    run_analysis_on_hf_dataset(
        hf_dataset,
        parsed_known_metadata,
        output_directory,
        numerical_fraction,
        get_correlations=get_correlations,
        get_predictability=get_predictability,
    )


@click.command()
@click.option(
    "--data_directory",
    help="path to main folder with all the data files. Note: only CSVs are currently supported",
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--output_directory",
    help="output directory for the report",
    required=True,
    type=click.Path(path_type=Path, exists=True),
)
@click.option(
    "--known_metadata",
    help="known metadata of the dataset",
    required=False,
    type=str,
    default="{}",
)
@click.option(
    "--numerical_fraction",
    help="fraction of numerical values to consider a feature numerical",
    default=0.9,
    type=float,
)
@click.option(
    "--get_predictability",
    help="whether to run a target predictability analysis on tabular features",
    default=False,
    type=float,
    is_flag=True,
)
@click.option(
    "--get_correlations",
    help="whether to run a pairwise feature correlation analysis on tabular features",
    default=False,
    type=float,
    is_flag=True,
)
def load_and_run_data_analysis_on_folder(
    data_directory: Path,
    output_directory: Path,
    known_metadata: str = "{}",
    numerical_fraction: float = 0.9,
    get_predictability: bool = False,
    get_correlations: bool = False,
) -> None:
    """Runs a basic data analysis script that generates a dataset report with metadata.

    Args:
        data_directory: path to main folder with all the data files. Note: only CSVs are currently supported.
        output_directory: output directory for the report.
        known_metadata: known metadata of the dataset.
        numerical_fraction: fraction of numerical values to consider a feature numerical. Defaults to 0.9.
    """
    parsed_known_metadata: Dict[str, Any] = ast.literal_eval(known_metadata)
    parsed_known_metadata["data_directory"] = str(data_directory)

    run_analysis_on_folder(
        data_directory,
        output_directory,
        numerical_fraction=numerical_fraction,
        get_correlations=get_correlations,
        get_predictability=get_predictability,
        known_metadata=parsed_known_metadata,
    )


if __name__ == "__main__":
    run_agent_cli()
