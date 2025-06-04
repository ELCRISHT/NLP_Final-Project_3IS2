# NLP_Final-Project_3IS2
# SMS Spam Collection NLP Analysis

This project performs a comprehensive Natural Language Processing (NLP) analysis on the SMS Spam Collection dataset. The analysis includes text statistics, word frequency analysis, text normalization, and semantic analysis using TF-IDF and clustering.

Table of Contents
- [Project Description](#project-description)
- [Setup and Installation](#setup-and-installation)
- [Dataset](#dataset)
- [Running the Code](#running-the-code)
- [Analysis Steps](#analysis-steps)
- [Results and Visualizations](#results-and-visualizations)
- [Dependencies](#dependencies)

Project Description

This notebook analyzes SMS messages to differentiate between "ham" (legitimate) and "spam" messages using various NLP techniques. The key steps include:
- Loading the Dataset:** Downloading and loading the SMS Spam Collection dataset.
- Basic Analysis:** Displaying dataset overview, label distribution, and sample messages.
- Text Statistics:** Calculating and visualizing message length, word count, average word length, and punctuation count for ham and spam messages.
- Word Frequency Analysis:** Identifying and visualizing the most frequent words in both ham and spam messages using word clouds and bar plots.
- Text Normalization:** Applying regular expressions to normalize text, including tokenizing specific patterns like emails, URLs, phone numbers, monetary amounts, percentages, and dates, and normalizing numbers and removing special characters.
- Semantic Analysis:** Using TF-IDF to vectorize the text, calculating semantic similarity, clustering messages based on semantic content, and visualizing the clusters and top terms.

Setup and Installation
This code is designed to run in a Google Colab environment or a Jupyter Notebook.
1.  Open the Notebook:Open the provided Python notebook file in Google Colab or your Jupyter environment.
2.  Run the Setup Cells: The first few code cells handle the installation of necessary libraries and the download of NLTK data and spaCy models. Run these cells first to ensure all dependencies are met.

Dataset
The project uses the [SMS Spam Collection dataset](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) from the UCI Machine Learning Repository. The notebook automatically downloads and loads this dataset. A fallback sample dataset is included in case the download fails.

Running the Code
Once the setup cells are executed, you can run the remaining cells sequentially to perform the analysis. Each cell performs a specific part of the NLP pipeline:
1.  Load Dataset:** Loads the SMS Spam Collection data into a pandas DataFrame.
2.  Basic Dataset Information: Prints summary statistics and sample messages.
3.  Calculate Text Statistics: Computes and displays descriptive statistics about the messages (length, word count, etc.) and generates box plots.
4.  Word Frequency Analysis: Calculates and displays the most frequent words and generates word clouds and bar plots.
5.  Text Normalization: Applies regular expressions for cleaning and tokenization, and prints sample results and statistics.
6.  Semantic Analysis: Performs TF-IDF vectorization, clustering, and generates visualizations for the semantic space and top terms.

To run each cell in Google Colab or Jupyter:
-   Select the cell and press `Shift + Enter`.
-   Alternatively, use the "Run cell" button that appears when you hover over the cell.
-   You can run all cells in order using the "Runtime" menu -> "Run all".

Analysis Steps
The notebook follows these main steps for the NLP analysis:
-   Data Loading and Inspection:** Understanding the structure and content of the dataset.
-   Exploratory Data Analysis (EDA):** Analyzing basic text statistics and word frequencies to gain initial insights into the differences between ham and spam.
-   Text Preprocessing:** Normalizing the text by handling specific patterns (emails, URLs, etc.) and cleaning special characters and numbers.
-   Semantic Analysis:** Using advanced techniques like TF-IDF and K-Means clustering to explore the underlying topics and similarity between messages.

Results and Visualizations
The notebook generates various outputs and visualizations throughout the process, including:
-   Dataset shape, columns, and data types.
-   Label distribution and percentages.
-   Sample ham and spam messages.
-   Tables of text statistics (mean, std, min, max) by label.
-   Box plots for message length, word count, average word length, and punctuation count.
-   Lists of top frequent words for ham and spam.
-   Word clouds for ham and spam messages.
-   Bar plots comparing the frequency of top words.
-   Normalization statistics (counts of extracted patterns).
-   Sample text before and after normalization.
-   TF-IDF matrix shape and vocabulary size.
-   Number of semantic clusters and key terms for each cluster.
-   Pie chart showing semantic cluster distribution.
-   2D scatter plot visualizing the semantic clusters using PCA.
-   Heatmap of semantic similarity between a sample of messages.
-   Bar plot of the top TF-IDF terms.

Dependencies
The code relies on the following Python libraries:

-   `pandas==1.5.3`
-   `numpy==1.23.5`
-   `nltk==3.8.1`
-   `spacy==3.5.2`
-   `textblob==0.17.1`
-   `scikit-learn==1.2.2`
-   `matplotlib==3.7.1`
-   `seaborn==0.12.2`
-   `wordcloud==1.8.2.2`
-   `plotly==5.13.1`
-   `python-Levenshtein==0.21.1` (Optional, though included in the install cell)

These libraries are automatically installed by the first code cell in the notebook.

