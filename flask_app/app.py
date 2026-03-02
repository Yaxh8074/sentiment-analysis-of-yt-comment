# main.py

import matplotlib
matplotlib.use('Agg')

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import io
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import mlflow
import joblib
import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import matplotlib.dates as mdates
import dagshub
import os

app = FastAPI(title="YouTube Sentiment Analysis API")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------------------------------------------------
# DagsHub MLflow Tracking Setup
# -------------------------------------------------------------------------
DAGSHUB_USERNAME = os.environ.get('DAGSHUB_USERNAME', 'Yaxh8074')
DAGSHUB_REPO = 'youtube-comment-analysis'

dagshub.init(
    repo_owner=DAGSHUB_USERNAME,
    repo_name=DAGSHUB_REPO,
    mlflow=True
)
# -------------------------------------------------------------------------
# AWS EC2 MLflow Tracking (commented out)
# mlflow.set_tracking_uri("http://ec2-54-196-109-131.compute-1.amazonaws.com:5000/")
# -------------------------------------------------------------------------

# -------------------------------------------------------------------------
# Pydantic Models
# -------------------------------------------------------------------------
class Comment(BaseModel):
    text: str
    timestamp: Optional[str] = None

class CommentsRequest(BaseModel):
    comments: List[str]

class CommentsWithTimestampRequest(BaseModel):
    comments: List[Comment]

class SentimentCount(BaseModel):
    sentiment_counts: dict

class SentimentDataRequest(BaseModel):
    sentiment_data: List[dict]

class WordCloudRequest(BaseModel):
    comments: List[str]

# -------------------------------------------------------------------------
# Preprocessing
# -------------------------------------------------------------------------
def preprocess_comment(comment: str) -> str:
    """Apply preprocessing transformations to a comment."""
    try:
        comment = comment.lower()
        comment = comment.strip()
        comment = re.sub(r'\n', ' ', comment)
        comment = re.sub(r'[^A-Za-z0-9\s!?.,]', '', comment)

        stop_words = set(stopwords.words('english')) - {'not', 'but', 'however', 'no', 'yet'}
        comment = ' '.join([word for word in comment.split() if word not in stop_words])

        lemmatizer = WordNetLemmatizer()
        comment = ' '.join([lemmatizer.lemmatize(word) for word in comment.split()])

        return comment
    except Exception as e:
        print(f"Error in preprocessing comment: {e}")
        return comment

# -------------------------------------------------------------------------
# Load Model & Vectorizer
# -------------------------------------------------------------------------
def load_model_and_vectorizer(model_name: str, model_version: str, vectorizer_path: str):
    # -------------------------------------------------------------------------
    # Alias-based loading (commented out - DagsHub may not support aliases)
    # model_uri = f"models:/{model_name}@{model_alias}"
    # -------------------------------------------------------------------------
    model_uri = f"models:/{model_name}/{model_version}"
    model = mlflow.pyfunc.load_model(model_uri)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
VECTORIZER_PATH = os.path.join(BASE_DIR, 'tfidf_vectorizer.pkl')

model, vectorizer = load_model_and_vectorizer(
    "yt_chrome_plugin_model",
    "1",
    VECTORIZER_PATH
)

# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------
@app.get("/")
def home():
    return {"message": "Welcome to the YouTube Sentiment Analysis API"}


@app.post("/predict")
def predict(request: CommentsRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        preprocessed = [preprocess_comment(c) for c in request.comments]
        transformed = vectorizer.transform(preprocessed)
        predictions = [str(p) for p in model.predict(transformed).tolist()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return [{"comment": c, "sentiment": s} for c, s in zip(request.comments, predictions)]


@app.post("/predict_with_timestamps")
def predict_with_timestamps(request: CommentsWithTimestampRequest):
    if not request.comments:
        raise HTTPException(status_code=400, detail="No comments provided")
    try:
        texts = [item.text for item in request.comments]
        timestamps = [item.timestamp for item in request.comments]

        preprocessed = [preprocess_comment(c) for c in texts]
        transformed = vectorizer.transform(preprocessed)
        predictions = [str(p) for p in model.predict(transformed).tolist()]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    return [{"comment": c, "sentiment": s, "timestamp": t}
            for c, s, t in zip(texts, predictions, timestamps)]


@app.post("/generate_chart")
def generate_chart(request: SentimentCount):
    try:
        sentiment_counts = request.sentiment_counts
        labels = ['Positive', 'Neutral', 'Negative']
        sizes = [
            int(sentiment_counts.get('1', 0)),
            int(sentiment_counts.get('0', 0)),
            int(sentiment_counts.get('-1', 0))
        ]
        if sum(sizes) == 0:
            raise ValueError("Sentiment counts sum to zero")

        colors = ['#36A2EB', '#C9CBCF', '#FF6384']

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
                startangle=140, textprops={'color': 'w'})
        plt.axis('equal')

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG', transparent=True)
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chart generation failed: {str(e)}")


@app.post("/generate_wordcloud")
def generate_wordcloud(request: WordCloudRequest):
    try:
        if not request.comments:
            raise HTTPException(status_code=400, detail="No comments provided")

        preprocessed = [preprocess_comment(c) for c in request.comments]
        text = ' '.join(preprocessed)

        wordcloud = WordCloud(
            width=800, height=400,
            background_color='black',
            colormap='Blues',
            stopwords=set(stopwords.words('english')),
            collocations=False
        ).generate(text)

        img_io = io.BytesIO()
        wordcloud.to_image().save(img_io, format='PNG')
        img_io.seek(0)

        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Word cloud generation failed: {str(e)}")


@app.post("/generate_trend_graph")
def generate_trend_graph(request: SentimentDataRequest):
    try:
        if not request.sentiment_data:
            raise HTTPException(status_code=400, detail="No sentiment data provided")

        df = pd.DataFrame(request.sentiment_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        df['sentiment'] = df['sentiment'].astype(int)

        sentiment_labels = {-1: 'Negative', 0: 'Neutral', 1: 'Positive'}
        monthly_counts = df.resample('M')['sentiment'].value_counts().unstack(fill_value=0)
        monthly_totals = monthly_counts.sum(axis=1)
        monthly_percentages = (monthly_counts.T / monthly_totals).T * 100

        for sentiment_value in [-1, 0, 1]:
            if sentiment_value not in monthly_percentages.columns:
                monthly_percentages[sentiment_value] = 0

        monthly_percentages = monthly_percentages[[-1, 0, 1]]

        plt.figure(figsize=(12, 6))
        colors = {-1: 'red', 0: 'gray', 1: 'green'}

        for sentiment_value in [-1, 0, 1]:
            plt.plot(
                monthly_percentages.index,
                monthly_percentages[sentiment_value],
                marker='o', linestyle='-',
                label=sentiment_labels[sentiment_value],
                color=colors[sentiment_value]
            )

        plt.title('Monthly Sentiment Percentage Over Time')
        plt.xlabel('Month')
        plt.ylabel('Percentage of Comments (%)')
        plt.grid(True)
        plt.xticks(rotation=45)
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=12))
        plt.legend()
        plt.tight_layout()

        img_io = io.BytesIO()
        plt.savefig(img_io, format='PNG')
        img_io.seek(0)
        plt.close()

        return StreamingResponse(img_io, media_type="image/png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend graph generation failed: {str(e)}")


# -------------------------------------------------------------------------
if __name__ == '__main__':
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)