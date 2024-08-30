# Load necessary libraries
import pandas as pd
import numpy as np
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv('path_to_your_dataset.csv')

# Example EDA: Sentiment Analysis
sid = SentimentIntensityAnalyzer()
df['sentiment'] = df['headline'].apply(lambda x: sid.polarity_scores(x)['compound'])

# Example EDA: Basic statistics
df['headline_length'] = df['headline'].apply(len)
print(df['headline_length'].describe())

# Example EDA: Publication trends
df['date'] = pd.to_datetime(df['date'])
df['day_of_week'] = df['date'].dt.day_name()
sns.countplot(data=df, x='day_of_week')
plt.show()

