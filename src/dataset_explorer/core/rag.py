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

"""Dataset retrieval based on the results of systematic automatic exploratory dataset analysis"""

# from langchain_community.document_loaders.csv_loader import CSVLoader
import json
import uuid
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
from langchain.docstore.document import Document
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import DistanceStrategy
from langchain_core.prompts import PromptTemplate
from loguru import logger

from dataset_explorer.models import create_llm
from dataset_explorer.models.embedding import LongTextHFEmbeddings


def create_vector_store_from_metadata(
    path_to_file: Path,
    output_directory: Optional[Path] = None,
    **kwargs,  # noqa:ARG001
) -> FAISS:
    """Gets a metadata summary path and creates a local vector store for Q&A.

    Args:
        path_to_file: Path to the metadata summary report.

    Raises:
        ValueError: if the metadata summary path points to an inexistent file.

    Returns:
        A FAISS vector store built from the descriptions of the dataset summary.
    """

    def _preprocess_metadata_for_vector_store(data_row: pd.Series, add_examples: bool = False):
        """Preprocess metadata to add to vector store to drop unneded fields and add examples.

        Args:
            data_row: data being added to the vectorstore.
            add_examples: wheter to add data examples. Defaults to False.

        Returns:
            the preprocessed metadata.
        """
        folder = "results/"
        file_name = str(Path(str(data_row["dataset_name"]).replace(" ", "")).with_suffix(""))
        file_json_path = Path(folder + file_name).joinpath(
            Path(data_row["dataset_name"] + "-examples.json")
        )

        if "description" in data_row:
            data_row = data_row.drop(columns=["description"])
        if "zenodo_info" in data_row:
            data_row = data_row.drop(columns=["zenodo_info"])
        if "file_type" in data_row:
            if data_row["file_type"] == "file":
                data_row["location"] = f"local file: {file_name}"
            elif data_row["file_type"] == "record":
                data_row["location"] = "zenodo"
                if data_row["zenodo_record_id"] is not None:
                    data_row["location"] += f" record id {data_row['zenodo_record_id']}"
            elif data_row["file_type"] == "hf_dataset":
                data_row["location"] = "hugging face datasets"

        if not add_examples:
            return data_row

        if file_json_path.exists():
            with file_json_path.open("r") as json_file:
                data_examples = json.load(json_file)
        else:
            print(file_json_path)
            return data_row
        data_row["dataset_examples"] = data_examples
        return data_row

    if path_to_file.suffix == ".csv":
        data = pd.read_csv(path_to_file)
    elif path_to_file.suffix == ".pkl":
        data = pd.read_pickle(path_to_file)
    else:
        raise ValueError("Metadata summary file not supported. Pass a CSV or a Pickle file.")

    data = data.drop(columns="Unnamed: 0") if "Unnamed: 0" in data.columns else data
    if "description" not in data.columns:
        raise ValueError(
            "Dataset description not present in metadata. Cannot create vector store based on descriptions."
        )

    embeddings = LongTextHFEmbeddings()

    documents = [
        Document(
            page_content=f"{row['description']}",
            metadata=_preprocess_metadata_for_vector_store(row).to_dict(),
            id=str(uuid.uuid4()),
        )
        for _, row in data.iterrows()
    ]
    vector_store = FAISS.from_documents(
        documents,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )
    save_dir = Path("./output")
    if output_directory is not None:
        if output_directory.exists():
            save_dir = output_directory
        else:
            logger.info("Output dir not found. Saving in ./output")
            Path.mkdir("./output", exist_ok=False)
    vector_store.save_local(str(save_dir.joinpath("faiss_vector_store")))
    return vector_store


def create_vector_store_from_json_file(
    path_to_file: Path | str, output_dir: Optional[Path] = None, nr_lines: Optional[int] = None
) -> FAISS:
    """Gets a JSON file path and creates a local vector store for Q&A.

    Args:
        path_to_file: Path to the json file with the metadata.

    Raises:
        ValueError: if the json path points to an inexistent file.

    Returns:
        A FAISS vector store built from the metadata of the JSON file.
    """
    if isinstance(path_to_file, str):
        path_to_file = Path(path_to_file)
    if path_to_file.suffix == ".json" or path_to_file.suffix == ".jsonl":
        data = pd.read_json(path_to_file)

        if nr_lines is not None:
            data = data.head(nr_lines)
    else:
        raise ValueError("File not supported. Pass a JSON file.")

    data["description"] = data["metadata"].apply(
        lambda metadata: metadata["description"] if "description" in metadata.keys() else None
    )
    data = data.dropna(subset=["description"])

    embeddings = LongTextHFEmbeddings()

    llm = create_llm()

    template = (
        "Create a description for a file based on the following information about it: title is {title},"
        "existing description is {existing_description}."
    )

    prompt = PromptTemplate.from_template(template)

    documents = [
        Document(
            page_content=llm.invoke(  # type: ignore
                prompt.format(title=row["title"], existing_description=row["description"])
            ).content,  # type: ignore
            metadata={
                "doi": row["doi"],
                "dataset_name": row["title"],
                "file_type": "record",
                "original_description": row["description"],
            },
            id=str(uuid.uuid4()),
        )
        for _, row in data.iterrows()
    ]

    vector_store = FAISS.from_documents(
        documents,
        embeddings,
        distance_strategy=DistanceStrategy.MAX_INNER_PRODUCT,
        normalize_L2=True,
    )
    save_dir = Path("./output")
    if output_dir is not None:
        if output_dir.exists():
            save_dir = output_dir
        else:
            logger.info("Output dir not found. Saving in ./output")
            Path.mkdir("./output")
    vector_store.save_local(str(save_dir.parent.joinpath("faiss_vector_store")))
    return vector_store


def load_vector_store_from_path(vector_store_path: Path) -> FAISS:
    """Gets the path to a FAISS folder and returns the loaded vector store.

    Args:
        vector_store_path: path to where is saved the faiss index file.
    Raises:
        ValueError: If the path does not exists.

    Returns:
        The loaded vector store.
    """
    if not vector_store_path.exists():
        raise ValueError("Invalid path to vector store.")
    embeddings = LongTextHFEmbeddings()
    return FAISS.load_local(
        str(vector_store_path), embeddings, allow_dangerous_deserialization=True
    )


def query_search(
    query: str, vector_store: Optional[FAISS] = None, vector_store_path: Optional[Path] = None
) -> Tuple[str, str]:
    """Responds to a query about the analyzed dataset with RAG starting from a FAISS a vector store.

    Args:
        query: Question to answer about the dataset.
        vector_store: vector store (FAISS) object. Defaults to None.
        vector_store_path: path to load the vector store from. Defaults to None.

    Returns:
        The answer to the query
    """

    if not vector_store:
        if vector_store_path:
            vector_store = load_vector_store_from_path(vector_store_path)
        else:
            raise ValueError("Need a vector store.")
    results = vector_store.similarity_search(query)
    return f"Query answer={results[0].page_content}", str(results[0].page_content)


def merge_indexes(indexes: List[Path], output_directory: Path | None = None) -> None:
    """Merge indexes of multiple FAISS vector stores.

    Args:
        indexes: list of paths to faiss.index files.
        output_directory: save directory. Defaults to None.
    """
    embeddings = LongTextHFEmbeddings()
    index_data = None
    for index_path in indexes:
        print("Index path: ", index_path)
        if index_data is None:
            index_data = FAISS.load_local(
                str(index_path), embeddings, allow_dangerous_deserialization=True
            )
        else:
            index_data.merge_from(
                FAISS.load_local(str(index_path), embeddings, allow_dangerous_deserialization=True)
            )
    if output_directory is None:
        logger.info("No output dir selected. Saving in ./output")
        Path.mkdir("./output", exist_ok=False)
        output_directory = Path("./output")
    FAISS.save_local(index_data, str(output_directory))
    return
