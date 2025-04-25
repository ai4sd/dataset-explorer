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

"""RAG agent tools for the metadata scouter."""

from pathlib import Path
from typing import Optional, Tuple

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from dataset_explorer.core.rag import (
    create_vector_store_from_json_file,
    create_vector_store_from_metadata,
    query_search,
)


def create_vector_store_from_metadata_wrapper(
    metadata_summary_file: str, nr_lines: Optional[int]
) -> Tuple[str, str]:
    """Tool wrapper to create a vector store from a metadata summary file .

    Args:
        metadata_summary_file: path to metadata summary file passed as string.

    Returns:
        Path to the location on disk of the FAISS vector store for query search.
    """
    metadata_summary_path = Path(metadata_summary_file)
    if metadata_summary_path.suffix == ".json":
        _ = create_vector_store_from_json_file(metadata_summary_path, nr_lines)
    else:
        _ = create_vector_store_from_metadata(metadata_summary_path)
    return (
        f"Vector store has been created in location: {str(metadata_summary_path.parent.joinpath('faiss_vector_store'))}",
        str(metadata_summary_path.parent.joinpath("faiss_vector_store")),
    )


def answer_query_from_vector_store_wrapper(
    query: str, vector_store_directory: str = ""
) -> Tuple[str, str]:
    """Tool wrapper to answer a query given the path to a vector store of datasets.

    Args:
        query: User query.
        vector_store_directory: Directory where the vector store is located. Defaults to output/faiss_vector_store.

    Returns:
        The answer to the query.
    """
    return query_search(query, vector_store_path=Path(vector_store_directory))


class QuerySearchOnDatasetFolderInput(BaseModel):
    """RAG Query search input class from dataset folder."""

    query: str = Field(description="question to ask about the dataset.")
    vector_store_path: Optional[Path] = Field(
        None, description="path where the vector store of the dataset is saved."
    )


class CreateVectorStoreInput(BaseModel):
    """Base input model for creating a vector store from file."""

    metadata_summary_file: str = Field(
        description="path to the metadata summary file that encompasses all the information about the datasets."
    )
    nr_lines: Optional[int] = Field(description="number of lines to use from the original dataset")


CREATE_RAG_VECTOR_STORE_TOOL = StructuredTool.from_function(
    func=create_vector_store_from_metadata_wrapper,
    name="CreateVectorStoreFromMetadataFile",
    description="Creates a vector store to search a dataset with RAG. It returns the path to the database.",
    args_schema=CreateVectorStoreInput,  # type:ignore
    return_direct=True,
    response_format="content_and_artifact",
)

ANSWER_QUERY_TOOL = StructuredTool.from_function(
    func=query_search,
    name="AnswerQueryOnDatasetFromVectorStore",
    description="Answers a user query about the dataset. To do it, it requires a vector store to search the dataset with RAG.",
    args_schema=QuerySearchOnDatasetFolderInput,  # type:ignore
    return_direct=True,
    response_format="content_and_artifact",
)

TOOLS = [
    CREATE_RAG_VECTOR_STORE_TOOL,
    ANSWER_QUERY_TOOL,
]
