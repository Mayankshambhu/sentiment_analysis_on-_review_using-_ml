import streamlit as st
import torch
from transformers import BertTokenizer, BertModel
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import json
import gzip
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import plotly.express as px
import shap
import io
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

class SentimentClassifier(nn.Module):
    def __init__(self, n_classes: int):
        super(SentimentClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.drop = nn.Dropout(p=0.3)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

@st.cache_resource
def load_model():
    model_path = 'sentiment_model_20240802_063916.pth'
    n_classes = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = SentimentClassifier(n_classes).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model, device

model, device = load_model()

@st.cache_resource
def load_tokenizer():
    tokenizer_path = 'sentiment_tokenizer_20240802_063916'
    return BertTokenizer.from_pretrained(tokenizer_path)

tokenizer = load_tokenizer()

st.markdown(
    "<h1 style='text-align: center; font-size: 40px; color: #FA8072;'>💬 Sentiment Analysis on Customer Reviews</h1>",
    unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; font-size: 20px; color: #4682B4;'>This app predicts the sentiment of customer reviews using a fine-tuned BERT model. 📊</p>",
    unsafe_allow_html=True)

st.markdown("""
<style>
@keyframes pop {
  0% { transform: scale(0.9); opacity: 0.7; }
  100% { transform: scale(1.0); opacity: 1.0; }
}

.sentiment-container {
  display: flex;
  justify-content: center;
  align-items: center;
  margin-top: 20px;
}

.sentiment {
  font-size: 24px;
  animation: pop 0.5s ease forwards;
}

.very-negative { color: #FF0000; }
.negative { color: #FF6347; }
.neutral { color: #FFA500; }
.positive { color: #32CD32; }
.very-positive { color: #008000; }

.tooltip {
  position: relative;
  display: inline-block;
  cursor: pointer;
  color: #007BFF;
}

.tooltip .tooltiptext {
  visibility: hidden;
  width: 220px;
  background-color: #333;
  color: #fff;
  text-align: center;
  border-radius: 6px;
  padding: 5px 0;
  position: absolute;
  z-index: 1;
  bottom: 125%;
  left: 50%;
  margin-left: -110px;
  opacity: 0;
  transition: opacity 0.3s;
}

.tooltip:hover .tooltiptext {
  visibility: visible;
  opacity: 1;
}

#reset-button {
  display: none;
  margin-top: 20px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
    <div style='background-color: #000000; padding: 10px; border-radius: 5px;'>
        <h3>How to Use This App:</h3>
        <ul>
            <li>Upload a file containing customer reviews or enter a review directly into the text area.</li>
            <li>Click the <span class='tooltip'>🔍 Analyze<span class='tooltiptext'>Analyzes the uploaded file or text review for sentiment.</span></span> button to start the sentiment analysis.</li>
            <li>If you uploaded a file with more than 10 reviews, visualizations such as heatmaps and word clouds will be displayed.</li>
            <li>For files with 10 or fewer reviews, individual sentiment analysis will be shown.</li>
        </ul>
        <p>Use the <strong>progress bar</strong> to track the analysis progress for large files.</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("📁 Upload a file (CSV, Excel, JSON, JSONL, JSONL.GZ):",
                                 type=["csv", "xlsx", "xls", "json", "jsonl", "gz"],
                                 help="Upload a file containing customer reviews.")

user_input = st.text_area("📝 Or, enter a customer review below:", "", height=150,
                          help="Enter your review here to analyze its sentiment.")

analysis_done = False

if uploaded_file and user_input.strip():
    st.error("⚠️ Please either upload a file or enter text manually, not both.")
else:
    analyze_button = st.button("🔍 Analyze")

    if analyze_button:
        analysis_done = True


        def analyze_review(review_text):
            if not review_text.strip():
                return "No review text provided", "🤔", "neutral", 0.0

            encoding = tokenizer.encode_plus(
                review_text,
                add_special_tokens=True,
                max_length=128,
                return_token_type_ids=False,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].to(device)
            attention_mask = encoding['attention_mask'].to(device)

            with torch.no_grad():
                outputs = model(input_ids, attention_mask=attention_mask)
                probs = F.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, dim=1)
                sentiment = preds.item()
                confidence = probs[0][sentiment].item() * 100

            sentiment_labels = [
                ('Very Negative', '😡', 'very-negative'),
                ('Negative', '😞', 'negative'),
                ('Neutral', '😐', 'neutral'),
                ('Positive', '🙂', 'positive'),
                ('Very Positive', '😍', 'very-positive')
            ]
            sentiment_text, sentiment_emoji, sentiment_class = sentiment_labels[sentiment]

            return sentiment_text, sentiment_emoji, sentiment_class, confidence


        def display_sentiment_analysis(review_text, sentiment_text, sentiment_emoji, sentiment_class, confidence):
            st.markdown(f"""
            <div class='sentiment-container'>
                <div class='sentiment {sentiment_class}'>
                    <p>{sentiment_emoji} {sentiment_text} - Confidence: {confidence:.2f}%</p>
                </div>
            </div>
            """, unsafe_allow_html=True)

        def plot_heatmap(data, column):
            sentiment_counts = data[column].value_counts()
            heatmap_data = sentiment_counts.reset_index()
            heatmap_data.columns = ['Sentiment', 'Count']
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(heatmap_data.set_index('Sentiment').T, annot=True, cmap="YlGnBu", fmt='d', ax=ax)
            ax.set_title('Sentiment Heatmap')
            st.pyplot(fig)

        def plot_interactive_bar(data, column):
            sentiment_counts = data[column].value_counts().reset_index()
            sentiment_counts.columns = ['Sentiment', 'Count']
            fig = px.bar(sentiment_counts,
                         x='Sentiment',
                         y='Count',
                         title='Interactive Sentiment Distribution')
            st.plotly_chart(fig)

        def generate_wordcloud(text):
            wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
            fig, ax = plt.subplots(figsize=(12, 8))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)

        @st.cache_resource
        def explain_prediction(text):
            explainer = shap.Explainer(model, tokenizer)
            encoding = tokenizer(text, return_tensors='pt', max_length=128, truncation=True, padding='max_length').to(
                device)

            with torch.no_grad():
                shap_values = explainer(encoding.input_ids, attention_mask=encoding.attention_mask)

            return shap_values


        def process_file(file):
            file_type = file.name.split('.')[-1].lower()

            if file_type == 'csv':
                data = pd.read_csv(file)
            elif file_type in ['xlsx', 'xls']:
                data = pd.read_excel(file)
            elif file_type == 'json':
                data = pd.read_json(file)
            elif file_type == 'jsonl':
                data = pd.read_json(file, lines=True)
            elif file_type == 'gz':
                with gzip.open(file, 'rt') as f:
                    data = pd.read_json(f, lines=True)
            else:
                raise ValueError("Unsupported file format.")

            return data


        def get_review_column(data):
            possible_columns = ['review', 'text', 'Test']
            for col in possible_columns:
                if col in data.columns:
                    return col
            raise ValueError("No 'review', 'text', or 'Test' column found in the data.")


        if uploaded_file:
            try:
                data = process_file(uploaded_file)

                if not data.empty:
                    review_column = get_review_column(data)

                    progress_bar = st.progress(0)

                    analysis_results = []
                    batch_size = 32
                    for i in range(0, len(data), batch_size):
                        batch = data[review_column][i:i + batch_size]
                        batch_results = [analyze_review(review) for review in batch]
                        analysis_results.extend(batch_results)
                        progress = (i + len(batch)) / len(data)
                        progress_bar.progress(int(progress * 100))

                    data['sentiment_text'], data['sentiment_emoji'], data['sentiment_class'], data['confidence'] = zip(
                        *analysis_results)

                    if len(data) > 10:
                        st.markdown("### Overall Sentiment Distribution")

                        st.markdown("#### Sentiment Heatmap")
                        plot_heatmap(data, 'sentiment_text')

                        st.markdown("#### Interactive Sentiment Distribution")
                        plot_interactive_bar(data, 'sentiment_text')

                        st.markdown("#### Word Cloud")
                        all_text = ' '.join(data[review_column].astype(str))
                        generate_wordcloud(all_text)
                    else:
                        st.markdown("### Individual Sentiment Analysis")
                        st.write(data[[review_column, 'sentiment_text', 'sentiment_emoji', 'confidence']])

                    csv = data.to_csv(index=False)
                    st.download_button(
                        label="Download results as CSV",
                        data=csv,
                        file_name="sentiment_analysis_results.csv",
                        mime="text/csv",
                    )



            except Exception as e:
                st.error(f"Error processing file: {str(e)}")
                st.stop()

        elif user_input.strip():
            if len(user_input) > 1000:
                st.warning("Review text is too long. Please limit to 1000 characters.")
                st.stop()

            sentiment_text, sentiment_emoji, sentiment_class, confidence = analyze_review(user_input)
            display_sentiment_analysis(user_input, sentiment_text, sentiment_emoji, sentiment_class, confidence)


            single_review_data = pd.DataFrame({
                'Review': [user_input],
                'Sentiment': [sentiment_text],
                'Confidence': [confidence]
            })
            csv = single_review_data.to_csv(index=False)
            st.download_button(
                label="Download result as CSV",
                data=csv,
                file_name="single_review_sentiment.csv",
                mime="text/csv",
            )

        st.markdown("""
        <button id="reset-button" onclick="window.location.reload()">Reset</button>
        """, unsafe_allow_html=True)