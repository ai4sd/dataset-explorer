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

"""Tests the rag retrieval and QA module."""

from pathlib import Path

from dataset_explorer.core.rag import create_vector_store_from_metadata, query_search


def test_create_vector_store_form_metadata():
    """Test function for RAG."""
    metadata_summary_path = Path("test_results/test-test_data_metadata_summary.csv")
    if metadata_summary_path.exists():
        vector_store = create_vector_store_from_metadata(metadata_summary_path)
        assert vector_store


def test_query_search():
    """Tests query search function on a cumulative report test path."""
    metadata_summary_path = Path("test_results/tests-test_data_metadata_summary.csv")
    vector_store = create_vector_store_from_metadata(metadata_summary_path)
    query_search("", vector_store=vector_store)
