# Sentiment_Analyzer

This project is a powerful and interactive sentiment analysis application that leverages **transformers**, **zero-shot learning**, and **VADER** techniques to analyze textual sentiment with high accuracy. It combines the strength of multiple NLP models into an ensemble system, accessible via an intuitive **Gradio interface**.

## 🚀 Features
- 🤖 **Transformer-based analysis** using RoBERTa (`cardiffnlp/twitter-roberta-base-sentiment-latest`)
- 🧠 **Zero-shot classification** via Facebook's BART (`facebook/bart-large-mnli`)
- 💬 **Rule-based sentiment analysis** using VADER from NLTK
- ⚖️ **Weighted ensemble model** to combine the three methods
- 🌐 **Web-based UI** powered by Gradio for real-time input and results
- 📊 Detailed breakdown of confidence scores per model

- ---

## 📁 File Structure

- `sentiment.py`: Main Gradio app with full UI and sentiment processing logic
- `sentiment_analyzer.py`: Backend sentiment analysis logic without Gradio interface
- `requirements.txt`: Required Python packages for running the application

---

## 🛠️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/your-username/your-repo-name.git
   cd your-repo-name

 ##  💻 Running the App
To launch the Gradio interface locally:

python sentiment.py (in bash/cmd/terminal)

This will open the application in your web browser at:
http://localhost:7860

## 🧪 Example Inputs

Try these example texts in the interface:

"I absolutely love this product! It's amazing and works perfectly."
"The service was terrible and the staff was rude. Never coming back!"
"The meeting was scheduled for 2pm in the conference room."

## 📦 Dependencies

From requirements.txt:

-transformers

-torch

-nltk

-gradio

-numpy

-pandas

## Make sure to also download the required NLTK corpora:
import nltk
nltk.download('vader_lexicon')
nltk.download('punkt')

## 📌 Notes

- The ensemble_analysis method assigns weights: 50% Transformer, 30% Zero-shot, 20% VADER.

- sentiment_analyzer.py can be used independently in other Python projects without UI.

  ## 🤝 Contributing
  
Contributions are welcome! Please fork the repo and submit a pull request.


