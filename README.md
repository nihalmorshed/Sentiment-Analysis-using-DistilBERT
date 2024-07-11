
# Sentiment Analysis Using DistilBERT Transformer Model

## Project Overview

This project demonstrates the implementation of a sentiment analysis model using the DistilBERT transformer. DistilBERT is a small, fast, cheap, and light Transformer model trained by distilling BERT base. It has 40% fewer parameters than google-bert/bert-base-uncased, and runs 60% faster while preserving over 95% of BERT’s performances as measured on the GLUE language understanding benchmark. The project showcases the powerful capabilities of transformer-based models in natural language processing tasks.

## Dataset

The dataset used in this project is the Twitter Airline Sentiment dataset, which contains tweets about airlines and their corresponding sentiment labels. The dataset has been preprocessed to include only the essential columns: `airline_sentiment` and `text`.

- **Source:** [Twitter Airline Sentiment Dataset](https://www.kaggle.com/crowdflower/twitter-airline-sentiment)
- **Columns Used:**
  - `airline_sentiment`: The sentiment of the tweet (positive, negative, neutral)
  - `text`: The actual text of the tweet

## Model and Training

The model used in this project is the DistilBERT transformer, a smaller and faster version of BERT. The training pipeline includes:

1. **Data Loading:** The dataset is loaded and preprocessed.
2. **Model Initialization:** DistilBERT model is initialized using the Hugging Face Transformers library.
3. **Training and Evaluation:** The model is trained on the dataset, and evaluation metrics such as ROC AUC Score and F1 Score are calculated.

## Results

The model achieved the following performance metrics:

- **ROC AUC Score:** 94.86%
- **F1 Score:** 92.86%

These results demonstrate the effectiveness of the DistilBERT model in sentiment analysis tasks.

## Usage

To run this project locally, follow these steps:

1. **Clone the Repository:**

```bash
git clone https://github.com/nihalmorshed/Sentiment-Analysis-using-DistilBERT
cd Sentiment-Analysis-using-DistilBERT
```

2. **Install the Required Libraries:**

```bash
pip install -r requirements.txt
```

3. **Download the Dataset:**

```bash
wget -nc https://lazyprogrammer.me/course_files/AirlineTweets.csv
```

4. **Run the Jupyter Notebook:**

Open the `Sentiment_Analysis_using_DistilBERT_transformer_model.ipynb` notebook and run all cells to see the implementation and results.

## Dependencies

- `numpy`
- `pandas`
- `matplotlib`
- `scikit-learn`
- `transformers`
- `torch`

## Acknowledgements

- The dataset is provided by CrowdFlower on Kaggle.
- The DistilBERT model is provided by the Hugging Face Transformers library.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
