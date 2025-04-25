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

"""Chunk embedder for the data scouter."""

from typing import List

import numpy as np
from langchain_community.embeddings import HuggingFaceEmbeddings
from pydantic import PrivateAttr
from transformers import BertTokenizer, PreTrainedTokenizer

from dataset_explorer.core.configuration import GEN_AI_SETTINGS


class LongTextHFEmbeddings(HuggingFaceEmbeddings):
    """Long context emedding creator."""

    _tokenizer: PreTrainedTokenizer = PrivateAttr()

    def __init__(self, model_name: str = GEN_AI_SETTINGS.text2vec_model_name):
        """Constructor for long text embedding.

        Args:
            model_name: HF model name to use for the embeddings. Defaults to GEN_AI_SETTINGS.text2vec_model_name.
        """
        super().__init__(model_name=model_name)
        self._tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

    def chunkize(self, text: str) -> List[List[int]]:
        """Gets an input text and returns it into chunks of tokens.

        Args:
            text: input text.

        Returns:
            list of chunks.
        """

        tokens: List[int] = self._tokenizer.encode(text)  # type: ignore
        return [
            tokens[i : i + GEN_AI_SETTINGS.text2vec_max_tokens_per_chunk]
            for i in range(0, len(tokens), GEN_AI_SETTINGS.text2vec_max_tokens_per_chunk)
        ]

    def embed(self, text: str) -> List[float]:
        """Gets an input text and returns an embedding optimized for retrieval.

        Args:
            text: input text.

        Returns:
            text embedding.
        """
        chunks = self.chunkize(text)

        chunk_embeddings = []
        for chunk in chunks:
            chunk_text = self._tokenizer.decode(chunk, skip_special_tokens=True)  # type: ignore
            chunk_embeddings.append(super().embed_query(chunk_text))
        # print('vector: ', np.mean(chunk_embeddings, axis=0)[:4])
        return list(np.mean(chunk_embeddings, axis=0))
