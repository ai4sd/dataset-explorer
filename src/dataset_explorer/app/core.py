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

"""Interactive data explorer app for Zenodo."""

import warnings
from typing import Any, List

import networkx as nx
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from annotated_text import annotated_text
from langchain_community.vectorstores import FAISS
from langchain_community.vectorstores.faiss import dependable_faiss_import
from langchain_core.documents.base import Document

from dataset_explorer.app.utils import add_ids, compute_graph_positions, scale, scatter_edge_traces
from dataset_explorer.core.configuration import DATA_ANALYSIS_SETTINGS, GEN_AI_SETTINGS
from dataset_explorer.core.utils import string_to_dict
from dataset_explorer.core.visualizer import iboxplot, scatterplot_all_features
from dataset_explorer.models.embedding import LongTextHFEmbeddings

warnings.filterwarnings("ignore")


@st.cache_data
def load_vector_store(faiss_directory: str) -> FAISS:
    """Loads vectorstore with caching.

    Args:
        faiss_directory: path to vectorstore as str.

    Returns:
        the loaded vectorstore.
    """
    embeddings = LongTextHFEmbeddings()
    return FAISS.load_local(faiss_directory, embeddings, allow_dangerous_deserialization=True)


@st.cache_data
def scatter_node_traces(_G: nx.Graph, hash: str) -> List[go.Scatter]:  # noqa: N803, ARG001
    """Creates scatterplot of nodes with caching.

    Args:
        _G: nx graph object
        hash: cached info. Needed to properly load the cache.

    Returns:
        List of scatterplots to superpose.
    """
    node_x = []
    node_y = []
    ids = []
    metadatas: List[Any] = []  # noqa: F841
    search_similarity = []
    node_attribute_to_color = []
    color_categorical = {
        val: col
        for col, val in zip(
            px.colors.qualitative.Plotly, ("file", "record", "search", "hf_dataset")
        )
    }
    node_attribute_to_text = []
    for node in _G.nodes():
        ids.append(node)

        metadata = vector_store.docstore.search(node).metadata  # type: ignore

        x, y = _G.nodes[node]["pos"]
        node_x.append(x)
        node_y.append(y)
        search_similarity.append(_G[node]["search"]["similarity"])

        node_attribute_to_color.append(color_categorical[metadata["file_type"]])
        # if "location" in metadata:
        #     node_attribute_to_text.append(str(metadata["location"]))
        # if "doi" in metadata:
        #     node_attribute_to_text.append(str(metadata["doi"]))
        if "dataset_name" in metadata:
            node_attribute_to_text.append(str(metadata["dataset_name"]))
        # if "file_format" in metadata:
        #     node_attribute_to_text.append(str(metadata["file_format"]))
        # if "creation_date" in metadata:
        #     node_attribute_to_text.append(str(metadata["creation_date"]))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        ids=ids,
        customdata=search_similarity,
        mode="markers",
        hoverinfo="text",
        hovertemplate="""
                <extra>Similarity: %{customdata}</extra>""",
        showlegend=False,
        marker=dict(
            color=node_attribute_to_color,
            size=10,
            line_width=2,
        ),
        text=node_attribute_to_text,
    )

    legend_traces = []
    for color_category, color in color_categorical.items():
        legend_traces.append(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=color_category,
                marker=dict(size=7, color=color),
            ),
        )
    return [node_trace, *legend_traces]


def main() -> None:
    """Main function to run the streamlit app"""
    # App initialization


with_search: bool = True
if not hasattr(st.session_state, "similarities"):
    st.session_state.similarities = None
st.title("Dataset explorer")

# Load vector store
with st.expander("Multi agent system details"):
    annotated_text((f"{GEN_AI_SETTINGS.language_model_name}", "llm"))
    annotated_text((f"{GEN_AI_SETTINGS.image2text_model_name}", "image2text"))
    annotated_text((f"{GEN_AI_SETTINGS.text2vec_model_name}", "text2vec"))
    for field in GEN_AI_SETTINGS.metadata_fields_for_prompt:
        annotated_text((f"{field}", "metadata"))
    for file_type in DATA_ANALYSIS_SETTINGS.supported_file_types_to_loader_map:
        annotated_text((f"{file_type}", "formats"))
    faiss_directory = st.text_input("Vector store location on disk", "output")

st.text("Specify path to vector store in the system details.")

vector_store = load_vector_store(faiss_directory)
vector_store = add_ids(vector_store)


# Search form and results
if with_search:
    with st.form("my_form"):
        query = st.text_area(
            "Enter a query for the dataset explorer",
            "Enter your query here.",
        )
        k = st.slider("Number of Documents to return", 1, vector_store.index.ntotal, 200)
        submitted = st.form_submit_button("Submit")
        if submitted:
            embedding = vector_store._embed_query(query)
            if vector_store._normalize_L2:
                faiss = dependable_faiss_import()
                st.session_state.embedding = faiss.normalize_L2(
                    np.array([embedding], dtype=np.float32)
                )
            else:
                st.session_state.embedding = embedding
            vector_store.add_documents([
                Document(
                    id="search",
                    page_content=query,
                    metadata={"file_type": "search", "doi": "Query"},
                )
            ])

            search_result = vector_store.similarity_search_with_score_by_vector(
                st.session_state.embedding,
                k,
            )
            st.session_state.query_text = query
            st.session_state.docstore_indices = [
                doc.id for doc, _ in search_result if doc.id is not None
            ]
            st.session_state.docstore_indices.append("search")
            reversed_index = {id_: idx for idx, id_ in vector_store.index_to_docstore_id.items()}

            st.session_state.vector_indices = [
                reversed_index[doc_id] for doc_id in st.session_state.docstore_indices
            ]
            st.session_state.reconstructed = vector_store.index.reconstruct_batch(
                st.session_state.vector_indices
            )
            similarities = st.session_state.reconstructed.dot(st.session_state.reconstructed.T)
            st.session_state.similarities = similarities

if st.session_state.similarities is None:
    message = st.chat_message("assistant")
    message.write("Hello! Enter a query to start.")

# Plotting the graph
if st.session_state.similarities is not None:
    with st.expander("Query pairwise similarities"):
        flattened = st.session_state.similarities[-1][:]
        flattened = flattened[:-1]

        fig_histogram = px.histogram(flattened)
        st.plotly_chart(
            fig_histogram,
            key="histogram",
        )
    similarities = st.session_state.similarities
    scaled_weights = scale(similarities, to_one=0.7)

    scaled_weights[:, -1] *= 10000
    scaled_weights[-1, :] *= 10000

    G, hash = compute_graph_positions(scaled_weights, nodelist=st.session_state.docstore_indices)
    nx.set_edge_attributes(
        G,
        nx.from_numpy_array(
            similarities, edge_attr="similarity", nodelist=st.session_state.docstore_indices
        ).edges,
    )

    top_k_edges = st.slider("Number of top similarities", -1, 20, 4, help="-1 is all, 0 is none.")

    fig = go.Figure(
        data=scatter_node_traces(G, hash) + scatter_edge_traces(G, hash, ["search"], top_k_edges),
        layout=go.Layout(
            title=dict(text="<br>Search results", font=dict(size=16)),
            showlegend=True,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            annotations=[
                dict(
                    text="Note that the position of nodes is dominated by attraction/repulsion from all other nodes.<br>The distance to the search query is not directly reflecting the similarity.",
                    showarrow=False,
                    xref="paper",
                    yref="paper",
                    x=0.005,
                    y=-0.002,
                )
            ],
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    event = st.plotly_chart(
        fig,
        key="plotly_vector_similarities_graph",
        on_select="rerun",
        selection_mode="points",
    )
    # event of point selection
    try:
        selected_point = event.selection["points"][0]  # type: ignore
        doc = vector_store.docstore.search(selected_point["id"])
        try:
            metadata = doc.metadata  # type: ignore
        except Exception as _:
            metadata = {"query_text": st.session_state.query_text, "dataset_name": "Your query"}

        with st.sidebar:
            st.title(metadata["dataset_name"])
            if "domain" in metadata:
                st.write(f"Domain: {metadata['domain']}")
            if "location" in metadata:
                st.write(metadata["location"])
            if "query_text" in metadata and metadata["dataset_name"] == "Your query":
                st.write(metadata["query_text"])
            if "doi" in metadata:
                if metadata["doi"] != "":
                    st.write(
                        # f'Zenodo link: [https://zenodo.org/records/{int(metadata["zenodo_record_id"])}](https://zenodo.org/records/{int(metadata["zenodo_record_id"])})'
                        f"Zenodo DOI: {metadata['doi']}"
                    )
            if "zenodo_record_id" in metadata:
                if metadata["zenodo_record_id"] != "":
                    st.write(
                        f"Zenodo link: [https://zenodo.org/records/{int(metadata['zenodo_record_id'])}](https://zenodo.org/records/{int(metadata['zenodo_record_id'])})"
                    )
            if isinstance(doc, Document) and metadata is not None:
                st.title("LLM description")
                st.write(doc.page_content)
                if "original_description" in metadata:
                    st.title("Original description:")
                    st.write(metadata["original_description"])

        # plotting statistics
        if "stats" in metadata:
            if isinstance(metadata["stats"], str):
                stats = string_to_dict(metadata["stats"])
                if "non_numerical_stats" in stats:
                    st.subheader("Non numeric features data analysis")
                    st.table(data=stats["non_numerical_stats"])
                if "numerical_stats" in stats:
                    st.subheader("Numeric features data analysis")
                    features = list(stats["numerical_stats"].keys())
                    df_stats = pd.DataFrame.from_dict(stats["numerical_stats"])
                    plot_obj = iboxplot(df_stats, features)
                    st.plotly_chart(plot_obj)
        # data visualizer
        if "dataset_examples" in metadata:
            st.subheader("Dataset examples")
            data = pd.DataFrame.from_records(metadata["dataset_examples"])
            st.dataframe(data)
            options = data.columns.tolist()
            feature_selection = st.multiselect(
                "Features to plot", [column for column in data.columns]
            )
            scaling = st.checkbox("Apply feature scaling")
            for feature in feature_selection:
                data[feature] = pd.to_numeric(data[feature], errors="coerce")
            numeric_data = data.loc[:, feature_selection].dropna()
            st.plotly_chart(scatterplot_all_features(numeric_data, scaling=scaling))

    except IndexError:
        pass

if __name__ == "__main__":
    main()
