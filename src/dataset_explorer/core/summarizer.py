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

"""Main functionalities for LLM-based summarization of content"""

from pathlib import Path
from typing import Union

from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.language_models.llms import BaseLLM
from langchain_core.prompts import PromptTemplate
from langchain_ibm import ChatWatsonx

from dataset_explorer.models import create_llm


def get_summary_map_reduce(text: str, llm: Union[BaseChatModel, BaseLLM, ChatWatsonx]) -> str:
    """Gets a summary of a long text.

    Args:
        text: input text.
        llm: language model.

    Returns:
        the summary string
    """
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], chunk_size=10000, chunk_overlap=500
    )  # chunk size = nr characters
    docs = text_splitter.create_documents([text])

    map_prompt = """
    Write a detailed summary of the following:
    "{text}"
    CONCISE SUMMARY:
    """
    map_prompt_template = PromptTemplate(template=map_prompt, input_variables=["text"])

    combine_prompt = """
    Write a detailed summary of the following text delimited by triple backquotes. Reason on the information in the paper, where possible, to understand the data. Be as detailed as possible.
    Return your response in bullet points which cover the key points.
    ```{text}```
    BULLET POINT SUMMARY:
    """
    combine_prompt_template = PromptTemplate(template=combine_prompt, input_variables=["text"])

    summary_chain = load_summarize_chain(
        llm=llm,
        chain_type="map_reduce",
        map_prompt=map_prompt_template,
        combine_prompt=combine_prompt_template,
    )

    output = summary_chain.invoke(docs)["output_text"]  # type: ignore
    return output


def precompute_paper_summaries(directory: Path) -> None:
    """Precomputes paper summaries.

    Args:
        directory: input directory of papers.
    """

    llm = create_llm()
    file_names = Path.iterdir(directory)
    for file_name in file_names:
        summary_file_name = file_name.replace(".txt", "-summary.txt")
        if Path.joinpath(directory, summary_file_name).exists():
            continue
        if "summary" in file_name:
            continue
        with Path.joinpath(directory, file_name).open("r") as file:
            pdf_content = file.read()

        print(f"Computing paper summary by MAP REDUCE: {file_name}\n")
        summary = get_summary_map_reduce(pdf_content, llm)

        with Path.joinpath(directory, summary_file_name).open("w") as summary_file:
            summary_file.write(summary)
