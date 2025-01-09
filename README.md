# Wine Rating Prediction Project

This project aims to predict wine ratings using machine learning models and fine-tuned language models. It includes data preprocessing, model training, and a Gradio interface for interactive predictions.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Project Structure](#project-structure)
5. [Data](#data)
6. [Results](#results)

---

## Project Overview

The goal of this project is to predict wine ratings based on descriptions and other features. The project includes:

- **Data Preprocessing**: Cleaning and preparing the wine dataset for modeling.
- **Model Training**: Training various machine learning models, including linear regression, random forests, and fine-tuned language models.
- **Gradio Interface**: An interactive web interface for predicting wine ratings and suggesting similar wines.

---

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/uliana0203/wine-project.git
   cd wine-project

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Set up environment variables:

    Create a `.env` file in the root directory and add your Hugging Face and OpenAI API keys:

    ```bash
    HF_TOKEN=your-huggingface-token
    OPENAI_API_KEY=your-openai-api-key

## Usage

1. Data Preprocessing
    Run the data preprocessing script to clean and prepare the dataset, if your want train your own model:

    ```bash
    python scripts/preprocessor.py

2. Gradio App
    Launch the Gradio app for real-time predictions with fine-tuned GPT-4o model or other OpenAI model:
    ```bash
    python scripts/gradio_interface.py

## Project Structure
    wine-project/
├── data/
│   ├── test_wine.pkl
│   ├── train_wine.pkl
│   └── wine_dataset.csv
├── notebooks/
│   ├── EDA wine dataset.ipynb
│   ├── llm_integration.ipynb
│   └── test_models.ipynb
├── scripts/
│   ├── preprocessor.py
│   ├── model.py
│   └── gradio_interface.py
├── requirements.txt
└── README.md

## Data

    The dataset used in this project is `alfredodeza/wine-ratings` from Hugging Face. It contains:

    1. Wine descriptions: Textual descriptions of wines.
    2. Ratings: Numeric ratings for each wine.
    3. Additional features: Region, variety, and year of production.

## Results

    # Performance Comparison Table:

| Model                          | Average Error | RMSLE  | Hit Rate |
|--------------------------------|---------------|--------|----------|
| Random Rating Predictor        | 5.36          | 0.07   | 4.7%     |
| Constant Rating Predictor      | 1.67          | 0.02   | 32.8%    |
| Linear Regression (Features)   | 1.67          | 0.02   | 33.9%    |
| BoW + Linear Regression        | 1.43          | 0.02   | 44.2%    |
| Word2Vec + Linear Regression   | 1.44          | 0.02   | 44.8%    |
| Word2Vec + SVR                 | 1.48          | 0.02   | 43.6%    |
| Word2Vec + Random Forest       | 1.38          | 0.02   | 47.1%    |
| GPT-4o-Mini (Pre-Trained)      | 2.03          | 0.03   | 14.3%    |
| GPT-4o Frontier (Pre-Trained)  | 2.00          | 0.03   | 16.1%    |
| Fine-Tuned GPT-4o-Mini         | 1.34          | 0.02   | 27.6%    |
| LLaMA 3.2 (via Ollama)         | 3.90          | 0.05   | 5.3%     |
    
