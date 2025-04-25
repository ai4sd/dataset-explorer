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

"""Interactive visualization functions for the metadata scouter visualization app"""

from typing import List

import pandas as pd
import plotly
import plotly.express as px


def iboxplot(
    data_statistics: pd.DataFrame,
    features: List[str],
) -> plotly.graph_objs._figure.Figure:
    """Creates an interactive boxplot from a data statistics dataframe.

    Args:
        data_statistics: data statistics to plot.
        features: features in the data statistics dataframe to add to the boxplot.
        save_path: path where to save the figure.
        title: title. Defaults to an empty string.
    """
    data_statistics = data_statistics[features]
    fig = px.box(data_statistics)
    return fig


def scatterplot_all_features(
    df: pd.DataFrame, scaling: bool = False
) -> plotly.graph_objs._figure.Figure:
    """Create a scatterplot with all the features from
    a dataframe. Before plotting, the features are
    normalized so that all of them have the same scale.

    Args:
        path_data: path to json file with data sample
    """

    if scaling:
        df = (df - df.min()) / (df.max() - df.min())
    fig = px.scatter(df)
    fig.update_traces(marker_size=40)
    return fig
