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

"""Functionalities for the interactive visualization app for the Zenodo scouter results"""

from pathlib import Path
from typing import List, Tuple

import networkx as nx
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from langchain_community.vectorstores import FAISS


@st.cache_data
def compute_graph_positions(adjacency_matrix: np.ndarray, nodelist: List) -> Tuple[nx.Graph, str]:
    """Compute graphs nodes positions based on adjacency matrix.

    Args:
        adjacency_matrix: input adjacency matrix.
        nodelist: input list of nodes.

    Returns:
        the graph object and hashing.
    """
    G = nx.from_numpy_array(adjacency_matrix, nodelist=nodelist)  # noqa: N806
    nx.set_node_attributes(G, nx.spring_layout(G, seed=7, iterations=100), name="pos")

    hash = nx.weisfeiler_lehman_graph_hash(G, node_attr="pos", edge_attr="weight")
    return G, hash


@st.cache_data
def scatter_edge_traces(_G: nx.Graph, hash: str, nodes=None, top_k_edges=-1) -> List[go.Scatter]:  # noqa: D103, N803, ARG001
    """Creates a scatterplot of edges with hashing.

    Args:
        _G: graph.
        hash: hash cache.
        nodes: nodes lilst. Defaults to None.
        top_k_edges: number of edges to plot. Defaults to -1.

    Returns:
        the graph objects to add as layers.
    """
    edge_x = []
    edge_y = []
    edge_marker_x = []
    edge_marker_y = []
    edge_marker_labeltext = None
    edge_marker_hovertext = []

    match top_k_edges:
        case -1:
            pass
        case _:
            for u, v, d in _G.edges(nbunch=nodes, data=True):
                xu, yu = _G.nodes[u]["pos"]
                xv, yv = _G.nodes[v]["pos"]
                edge_x.append((xu, xv))
                edge_y.append((yu, yv))
                edge_marker_x.append((xu + xv) / 2)
                edge_marker_y.append((yu + yv) / 2)
                edge_marker_hovertext.append(f"similarity={d['similarity']} weight={d['weight']}")

    if top_k_edges > 0:
        edge_marker_hovertext, edge_x, edge_y, edge_marker_x, edge_marker_y = zip(  # type: ignore
            *sorted(
                zip(edge_marker_hovertext, edge_x, edge_y, edge_marker_x, edge_marker_y),
                key=lambda five_tuple: five_tuple[0],
                reverse=True,
            )[1 : top_k_edges + 1]  # first top edge is self loop, to be ignored
        )
        edge_marker_labeltext = list(range(1, top_k_edges + 1))

    edge_trace = go.Scatter(
        x=[item for edge in ((u, v, None) for u, v in edge_x) for item in edge],
        y=[item for edge in ((u, v, None) for u, v in edge_y) for item in edge],
        line=dict(width=0.1, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    edge_marker_trace = go.Scatter(
        x=edge_marker_x,
        y=edge_marker_y,
        marker=dict(color="rgb(125,125,125)", size=1),
        hoverinfo="text",
        hovertext=edge_marker_hovertext,
        text=edge_marker_labeltext,
        mode="markers+text",
    )
    return [edge_trace, edge_marker_trace]


def read_markdown_file(markdown_file: str) -> str:
    """Reads markdown report file. - unused

    Args:
        markdown_file: path to markdown file as string.

    Returns:
        the md text.
    """
    return Path(markdown_file).read_text()


def add_ids(vector_store: FAISS):
    """Adds unique IDs to the entries in the vector store.

    Args:
        vector_store: vector store object.

    Returns:
        the vector store after indexing.
    """
    # TODO: See below. add way to track files within same record
    # record_id_to_id : Dict[int, List[int]] = defaultdict(list)
    some_doc_id = vector_store.index_to_docstore_id[0]
    some_doc = vector_store.docstore.search(some_doc_id)

    if some_doc.id is None:  # type: ignore
        # This patching works specifically for the InMemoryDocstore and uses its implementation details
        for id, doc in vector_store.docstore._dict.items():  # type: ignore
            # TODO: see below. Connected to first todo.
            # record_id = vector_store.docstore.metadata['zenodo_record_id']
            doc.id = id
    return vector_store


def scale(x, to_zero=0.0, to_one=0.56):  ## 0.56 0.91
    """Scales pairwise distances for better visualization.

    Args:
        x: data.
        to_zero: desired zero value. Defaults to 0.0.
        to_one: desired one value. Defaults to 0.56.

    Returns:
        the data rescaled between to zero and one.
    """
    return (x - to_zero) / (to_one - to_zero)


def plotly_node_selection_callback(node):
    """Writes the details of the selected node.

    Args:
        node: selected note.
    """
    st.write(node)
