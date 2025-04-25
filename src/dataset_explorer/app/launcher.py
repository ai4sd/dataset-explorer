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

"""Streamlit app launcher."""

import sys
from pathlib import Path

from loguru import logger


def main():
    """Run streamlit"""
    try:
        import streamlit.web.cli as stcli
    except ImportError:
        logger.info("Streamlit is not installed. Run [uv pip install streamlit].")
        sys.exit(1)

    package_dir = Path(__file__).parent.absolute()
    app_path = package_dir / "core.py"

    if not app_path.exists():
        logger.info(f"Error: {app_path} not found.")
        sys.exit(1)

    sys.argv = [
        "streamlit",
        "run",
        str(app_path),
        "--browser.gatherUsageStats=false",
        "--server.runOnSave=false",
        "--server.fileWatcherType=none",
    ]
    stcli.main()


if __name__ == "__main__":
    main()
