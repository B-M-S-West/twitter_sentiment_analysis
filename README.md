# Twitter Sentiment Analysis

A toolkit for analyzing and visualizing sentiment in tweets using classical machine learning and deep learning. Includes interactive notebooks, a GUI for sentiment visualization, and utility functions for image-based sentiment representation.

## Features

- Classical ML sentiment analysis (Naive Bayes, SVM, Logistic Regression)
- Deep learning sentiment analysis using HuggingFace Transformers 
- Interactive GUI for visualizing sentiment as expressive images
- Utility functions for color conversion and sentiment image generation

## Project Structure

```
twitter_sentiment_analysis/
├── data/                         # Dataset folder (ignored by git)
├── src/
│   └── interactive_sentiment_analysis/
│       ├── ctk/
│       │   └── ctk_image_display.py
│       ├── utils.py
│       └── sentiment_pipeline.py
├── twitter_sentiment.py          # Marimo notebook
├── pyproject.toml               
└── .python-version              
```

## Installation

### Option 1: Using pip

1. Clone the repository and change into the directory:
```bash
git clone https://github.com/yourusername/twitter_sentiment_analysis.git
cd twitter_sentiment_analysis
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -e .
```

### Option 2: Using uv (Recommended)

1. Clone the repository and change into the directory:
```bash
git clone https://github.com/yourusername/twitter_sentiment_analysis.git
cd twitter_sentiment_analysis
```

2. Install uv if you haven't already:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

3. Create a virtual environment and install dependencies:
```bash
uv venv
source .venv/bin/activate
uv pip install -e .
```

Note: uv provides faster dependency resolution and installation compared to pip. It's particularly useful for this project due to the machine learning dependencies.

## Usage

### Classical Sentiment Analysis with Marimo

1. Download the [Sentiment140 dataset](https://www.kaggle.com/datasets/kazanova/sentiment140) and place `training.1600000.processed.noemoticon.csv` in the `data/` folder

2. Run the Marimo notebook:
```bash
python twitter_sentiment.py
```
Or using uv you can run using the following:
```bash
uv run twitter_sentiment.py
```
Or to edit the file and run in the code blocks:
```bash
uv run marimo edit twitter_sentiment.py
```

The Marimo notebook provides an interactive interface where you can:
- Load and preprocess the Sentiment140 dataset
- Train three different models:
  - Bernoulli Naive Bayes
  - Support Vector Machine (SVM)
  - Logistic Regression
- View accuracy scores and classification reports
- Test the models with sample tweets in real-time

### Interactive GUI with CustomTkinter and Marimo

The project provides a custom GUI implementation using CustomTkinter for real-time sentiment visualization. This can also be done using a marimo script. Here is how to run them:
CustomTkinter GUI:
```
PYTHONPATH=src uv run src/interactive_sentiment_analysis/ctk/ctk_app.py
```
marimo webpage
```
PYTHONPATH=src uv run marimo run src/interactive_sentiment_analysis/marimo/marimo_app.py
```
Had to set the PYTHONPATH using the following as had issues with importing my modules. May need a better file structure for working with the projects in uv:
```
PYTHONPATH=src
```

Key Components:
- `CTkImageDisplay`: A custom widget that handles OpenCV image display
- `create_sentiment_image`: Generates an expressive face based on sentiment (-1 to +1)
- `SentimentAnalysisPipeline`: Wraps HuggingFace's transformers for sentiment analysis

Features:
- Real-time sentiment visualization
- Smooth image updates with frame queueing
- Automatic resizing while maintaining aspect ratio
- Support for both light and dark themes

## Modules

- **ctk_image_display.py**: CustomTkinter widget for OpenCV image display
- **utils.py**: Color conversion and sentiment image generation
- **sentiment_pipeline.py**: HuggingFace sentiment analysis pipeline wrapper

## Development

- Python 3.13
- Dependencies listed in `pyproject.toml`

## Acknowledgements

- [Sentiment140 Dataset](https://www.kaggle.com/datasets/kazanova/sentiment140)
- [HuggingFace Transformers](https://huggingface.co/transformers/)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
- [Marimo](https://marimo.io/)
- 