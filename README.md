
  # 🧠 Spam vs Ham Classifier using NLP & Machine Learning

  This project classifies SMS messages as **Spam 🚫** or **Ham ✔️ (not spam)** using Natural Language Processing and Machine Learning.  
  The focus is on text preprocessing, vectorization, model comparison, and understanding how representation impacts performance.

  ---

  ## 📌 Project Overview
  Raw text messages are converted into numerical vectors using:
  - Bag-of-Words (2-gram)
  - TF-IDF
  - Word2Vec (Google News 300d pretrained)
  - Custom-trained Word2Vec + Average Embeddings

  These representations are trained across ML models to observe performance variations.

  ---

  ## 📁 Dataset
  **SMS Spam Collection Dataset**  
  → Format: `label, message`  
  → Labels: `"spam"` or `"ham"`

  ---

  ## 🧠 NLP Pipeline

  Raw Text  
  ↓  
  Cleaning & Noise Removal  
  ↓  
  Tokenization + Stopwords Removal  
  ↓  
  Stemming / Lemmatization  
  ↓  
  Vectorization (BoW / TF-IDF / Embeddings)  
  ↓  
  Model Training  
  ↓  
  Evaluation & Results

  ---

  ## 🛠️ Features Implemented
  - Regex-based text cleaning
  - Stopword removal (NLTK)
  - Stemming & Lemmatization
  - Bag-of-Words (binary + n-gram)
  - TF-IDF vectorization
  - Word2Vec embeddings (pretrained + custom-trained)
  - Model training + classification report

  ---

  ## 🤖 Models Used
  - Multinomial Naive Bayes
  - Random Forest Classifier

  ---

  ## 📊 Model Performance (Results)
  - **Naive Bayes + BoW (2-gram):** ~97%  
  - **Naive Bayes + TF-IDF:** ~98%  
  - **Random Forest + TF-IDF:** **98%+ (best)**  
  - **Random Forest + Custom Word2Vec:** ~93–95%

  **Conclusion:** `TF-IDF + RandomForest` performed the best overall.

  ---

  ## 🛠️ Tech Stack
  - Python
  - Pandas, NumPy
  - NLTK (processing)
  - Scikit-learn (ML models)
  - Gensim (Word2Vec)
  - Google Colab / Jupyter Notebook

  ---

  ## 🚀 How to Run
  Install required libraries:
    pip install nltk scikit-learn gensim pandas numpy

  Run the notebook or .py file to train & evaluate models.

  *(Training automatically prints accuracy & classification report)*

  ---

  ## 🎯 Learning Outcomes
  ✔️ NLP preprocessing foundations  
  ✔️ Vectorization → performance relationship  
  ✔️ Word embeddings & averaging intuition  
  ✔️ How ML pipelines operate in text classification  

  ---

  ## 👩‍💻 Author
  **Shreya Gupta**  
  Aspiring AI/ML Engineer | NLP Learner  

  ---

  ## ✨ Closing Note
  > **Words can lie. Algorithms don't.**  
  > Before Transformers & GenAI — **NLP is the foundation.** 🚀
