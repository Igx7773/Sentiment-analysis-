# Sentiment-analysis-
#bert_emotion_classifiers
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from transformers import DataCollatorWithPadding
from datasets import Dataset

# Load dataset
df = pd.read_csv('twitter_training.csv', header=None)
df.columns = ['tweet_id', 'entity', 'label', 'text']
df = df[df['label'].isin(['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust'])]

# Map labels to integers
label2id = {label: idx for idx, label in enumerate(df['label'].unique())}
id2label = {idx: label for label, idx in label2id.items()}
df['label_id'] = df['label'].map(label2id)

# Split data
train_texts, val_texts, train_labels, val_labels = train_test_split(
    df['text'].tolist(), df['label_id'].tolist(), test_size=0.2, random_state=42)

# Tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

train_encodings = tokenizer(train_texts, truncation=True, padding=True)
val_encodings = tokenizer(val_texts, truncation=True, padding=True)

# Hugging Face Dataset
train_dataset = Dataset.from_dict({**train_encodings, 'label': train_labels})
val_dataset = Dataset.from_dict({**val_encodings, 'label': val_labels})

# Model
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',
    num_labels=len(label2id),
    id2label=id2label,
    label2id=label2id
)

# Training setup
training_args = TrainingArguments(
    output_dir='./results',
    evaluation_strategy='epoch',
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=3,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=10
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
)

# Train and Evaluate
trainer.train()
preds = trainer.predict(val_dataset)
pred_labels = torch.argmax(torch.tensor(preds.predictions), axis=1)

# Evaluation
print(classification_report(val_labels, pred_labels, target_names=label2id.keys()))

#model training
import joblib

# Save model
joblib.dump(model, 'sentiment_model.pkl')

# For tokenizer/BERT:
tokenizer.save_pretrained('tokenizer/')
model.save_pretrained('model/')

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

df = pd.read_csv("Emotion_Dataset.csv")
print(df.head())

sns.countplot(x='emotion', data=df)
plt.title("Emotion Counts")
plt.show()

import nltk
nltk.download('stopwords')

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    words = [ps.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)
df['cleaned'] = df['sentence'].apply(clean_text)

cv = CountVectorizer()
X = cv.fit_transform(df['cleaned'])
y = df['emotion']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = MultinomialNB()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))

for i in range(5):
    print("Text:", df['sentence'].iloc[i]) # Changed 'text' to 'sentence'
    print("Predicted Emotion:", model.predict(cv.transform([df['cleaned'].iloc[i]]))[0])

pandas
numpy
scikit-learn
transformers
torch
flask  # or fastapi for deployment
joblib  # if saving models
streamlit

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
df = pd.read_csv("twitter.training.csv")
print(df.head())
sns.countplot(x='emotion', data=df)
plt.title("Emotion Counts")
plt.show()
import nltk
nltk.download('stopwords')

import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def clean_text(text):
    text = re.sub(r'[^a-zA-Z]', ' ', text.lower())
    words = [ps.stem(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)
df['cleaned'] = df['sentence'].apply(clean_text)
cv = CountVectorizer()
X = cv.fit_transform(df['cleaned'])
y = df['emotion']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = MultinomialNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
for i in range(5):
    print("Text:", df['sentence'].iloc[i]) # Changed 'text' to 'sentence'
    print("Predicted Emotion:", model.predict(cv.transform([df['cleaned'].iloc[i]]))[0])

import streamlit as st
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# Load model and tokenizer
@st.cache_resource
def load_model():
    tokenizer = BertTokenizer.from_pretrained('tokenizer/')
    model = BertForSequenceClassification.from_pretrained('model/')
    return tokenizer, model

tokenizer, model = load_model()

# Streamlit UI
st.title("Emotion Detection from Tweets")
st.write("Enter a tweet or short text to detect its emotion.")

user_input = st.text_area("Your text here")

if st.button("Analyze"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        inputs = tokenizer(user_input, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            outputs = model(**inputs)
        probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
        emotion_id = torch.argmax(probs, dim=1).item()
        emotions = ['joy', 'anger', 'sadness', 'fear', 'surprise', 'disgust']
        st.success(f"**Detected Emotion:** {emotions[emotion_id]} ({probs.max().item()*100:.2f}% confidence)")


 "cells": [
  {
   "cell_type": "markdown",
   "id": "e81f174c",
   "metadata": {},
   "source": [
    "# Twitter Sentiment Analysis - EDA\n",
    "This notebook performs univariate and multivariate analysis on the Twitter sentiment dataset."
   ]
  },
  {
   "cell_type": "markdown",
   "id": "38c90920",
   "metadata": {},
   "source": [
    "## Univariate Analysis\n",
    "### Sentiment Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5493bee9",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "\n",
    "sns.countplot(x='sentiment', data=df)\n",
    "plt.title('Sentiment Distribution')\n",
    "plt.xlabel('Sentiment')\n",
    "plt.ylabel('Tweet Count')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3a8f5bab",
   "metadata": {},
   "source": [
    "### Tweet Length Distribution"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "c9b2c942",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df['tweet_length'] = df['clean_tweet'].apply(len)\n",
    "\n",
    "plt.figure(figsize=(10, 5))\n",
    "sns.histplot(df['tweet_length'], kde=True, bins=40)\n",
    "plt.title('Distribution of Tweet Lengths')\n",
    "plt.xlabel('Tweet Length')\n",
    "plt.ylabel('Frequency')\n",
    "plt.show()\n",
    "\n",
    "sns.boxplot(x='sentiment', y='tweet_length', data=df)\n",
    "plt.title('Tweet Length by Sentiment')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "bfea4e2a",
   "metadata": {},
   "source": [
    "## Bivariate/Multivariate Analysis\n",
    "### Word Count vs Sentiment"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "1d8227e5",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "df['word_count'] = df['clean_tweet'].apply(lambda x: len(x.split()))\n",
    "\n",
    "sns.boxplot(x='sentiment', y='word_count', data=df)\n",
    "plt.title(\"Word Count Distribution by Sentiment\")\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3040b1a8",
   "metadata": {},
   "source": [
    "### Correlation Matrix of Numeric Features"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "dcc314d8",
   "metadata": {},
   "outputs": [],
   "source": [
    "\n",
    "import numpy as np\n",
    "\n",
    "corr_matrix = df[['tweet_length', 'word_count']].corr()\n",
    "\n",
    "sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')\n",
    "plt.title('Correlation Matrix of Numeric Features')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7a53d071",
   "metadata": {},
   "source": [
    "## Insights Summary\n",
    "\n",
    "- **Sentiment Distribution**: Distribution of sentiment classes helps understand class imbalance.\n",
    "- **Tweet Length & Word Count**: These features vary with sentiment and may impact classification.\n",
    "- **Correlations**: Tweet length and word count are strongly correlated as expected.\n",
    "- **Feature Influence**: Textual features (via TF-IDF) are primary drivers of sentiment classification, with tweet length and word count potentially serving as supporting features.\n"
   ]
  }
 ],
 "metadata": {},
 "nbformat": 4,
 "nbformat_minor": 5
}

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from color_detector import load_colors, get_color_name

st.title("🎨 Color Detection Tool")
colors_df = load_colors("colors.csv")

uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "png", "jpeg"])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Click to detect color", use_column_width=True)
    image_np = np.array(image.convert('RGB'))
    click = st.image(image_np)
    if "click_x" not in st.session_state:
        st.session_state.click_x = st.session_state.click_y = 0
    def on_click(event):
        st.session_state.click_x = int(event.xdata)
        st.session_state.click_y = int(event.ydata)
    # Streamlit currently does not support click callbacks directly. 
    # To handle clicks, you may need to use OpenCV + local interface or use `streamlit-drawable-canvas`.
    st.write("Use OpenCV window for color detection (workaround):")
    if st.button("Open OpenCV Window"):
        img = cv2.imread(uploaded_file.name)
        img = cv2.resize(img, (600, 400))
        def show_color(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                b, g, r = img[y, x]
                color_name = get_color_name(r, g, b, colors_df)
                display = img.copy()
                cv2.rectangle(display, (20, 20), (300, 60), (int(b), int(g), int(r)), -1)
                cv2.putText(display, f"{color_name} ({r},{g},{b})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.imshow("Color Detection", display)
       cv2.namedWindow("Color Detection")
        cv2.setMouseCallback("Color Detection", show_color)
        cv2.imshow("Color Detection", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
  import pandas as pd

def load_colors(file_path):
    return pd.read_csv(file_path)

def get_color_name(R, G, B, df):
    min_distance = float('inf')
    closest_color = "Unknown"
    for _, row in df.iterrows():
        dist = abs(R - row["R"]) + abs(G - row["G"]) + abs(B - row["B"])
        if dist < min_distance:
            min_distance = dist
            closest_color = row["color_name"]
    return closest_color
      


