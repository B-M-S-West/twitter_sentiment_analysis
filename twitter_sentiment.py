import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from wordcloud import WordCloud
    import re
    import zipfile
    import requests
    from io import BytesIO
    import os

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

    return (
        BytesIO,
        LinearSVC,
        LogisticRegression,
        MultinomialNB,
        TfidfVectorizer,
        WordCloud,
        accuracy_score,
        classification_report,
        confusion_matrix,
        mo,
        os,
        pd,
        plt,
        re,
        requests,
        sns,
        train_test_split,
        zipfile,
    )


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Load dataset
    The first stage is to load the dataset. This is the Sentiment140 dataset from kaggle.
    """
    )
    return


@app.cell
def _(BytesIO, os, pd, requests, zipfile):
    # Define the file path
    file_path = 'data/training.1600000.processed.noemoticon.csv'

    # Check if file already exists
    if os.path.exists(file_path):
        print("File already exists, loading from local storage...")
        df = pd.read_csv(file_path)
        # The saved file should already have 'text' and 'polarity' columns
    else:
        print("File not found, downloading...")
        # Make sure the 'data' directory exists
        os.makedirs('data', exist_ok=True)
    
        # Download and extract
        url = "https://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
        response = requests.get(url)
        zip_file = zipfile.ZipFile(BytesIO(response.content))
        csv_file = zip_file.open('training.1600000.processed.noemoticon.csv')
    
        # Load into DataFrame
        df = pd.read_csv(csv_file, encoding='latin-1', header=None)
        df.columns = ['polarity', 'id', 'date', 'query', 'user', 'text']
    
        # Select only the columns we need
        df = df[['text', 'polarity']]
    
        # Save the filtered DataFrame
        df.to_csv(file_path, index=False)
        print("File downloaded and saved!")

    # Check what columns we actually have
    print("Columns:", df.columns.tolist())
    df.head()
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Keep positive and negative sentiments
    Remove the neutral tweets where polarity is 2. Then map labels so that 0 represents negative and then assign 1 for positive. Then calculate the number of positive and negative left in the dataset.
    Simple function to convert all text to lowercase so it is consistent and remove unwanted content like urls, mentions, hashtags.
    """
    )
    return


@app.cell
def _(df, re):
    def preprocess_text(text):
        text = text.lower()
        text = re.sub(r'http\S+', '', text)
        text = re.sub(r'@\w+', '', text)
        text = re.sub(r'#\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text


    df['text'] = df['text'].apply(preprocess_text)
    df['polarity'] = df['polarity'].map({4: 1, 2: 0, 0: -1})
    print(df['polarity'].value_counts())
    print(df.head())

    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Train Test Split
    This splits the clean_text and polarity into train and testing sets using an 80/20 split.
    """
    )
    return


@app.cell
def _(df, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        df['text'],
        df['polarity'],
        test_size=0.2,
        random_state=42
    )

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    return X_test, X_train, y_test, y_train


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Vectorisation
    This creates a TF IDF vectoriser that converts the text into numerical features.
    """
    )
    return


@app.cell
def _(TfidfVectorizer, X_test, X_train):
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1,2))

    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    print("TF-IDF shape (train):", X_train_tfidf.shape)
    print("TF-IDF shape (test):", X_test_tfidf.shape)
    return X_test_tfidf, X_train_tfidf, vectorizer


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Train Multinomial Naive Bayes model
    Train a Multinomial Naive Bayes classifier on the TF IDF features from the training data. It predicts sentiments for the test data and then prints the accuracy and a detailed classification report.
    """
    )
    return


@app.cell
def _(
    MultinomialNB,
    X_test_tfidf,
    X_train_tfidf,
    accuracy_score,
    classification_report,
    confusion_matrix,
    y_test,
    y_train,
):
    mnb = MultinomialNB()
    mnb.fit(X_train_tfidf, y_train)

    mnb_pred = mnb.predict(X_test_tfidf)

    print("Multinomial Naive Bayes Accuracy:", accuracy_score(y_test, mnb_pred))
    print("\nMultinomial Confusion Matrix:\n", confusion_matrix(y_test, mnb_pred))
    print("\nMultinomialNB Classification Report:\n", classification_report(y_test, mnb_pred))
    return (mnb_pred,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Dashboard for Data Visualisation
    Simple dashboard to look at the sentiment distribution, word cloud, and confusion matric

    1. Sentiment Distribution — A bar chart showing the number of positive, neutral, and negative tweets.
    2. Word Cloud — A visual representation of the most common words in the dataset.
    3. Sentiment Pie Chart — A pie chart showing the percentage distribution of sentiment categories.
    4. Confusion Matrix — A heatmap showing the performance of the model’s predictions.
    """
    )
    return


@app.cell
def _(WordCloud, confusion_matrix, df, mnb_pred, plt, sns, y_test):
    def create_dashboard():
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
        # Sentiment Distribution
        sns.countplot(x='polarity', data=df, ax=axes[0, 0])
        axes[0, 0].set_title('Sentiment Distribution')
    
        # Word Cloud
        text = ' '.join(df['text'])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(text)
        axes[0, 1].imshow(wordcloud, interpolation='bilinear')
        axes[0, 1].axis('off')
        axes[0, 1].set_title('Word Cloud')
    
        # Sentiment Pie Chart
        sentiment_counts = df['polarity'].value_counts()
        # Check the unique values in the 'polarity' column
        labels = sentiment_counts.index.tolist() 
        axes[1, 0].pie(sentiment_counts, labels=labels, autopct='%1.1f%%', startangle=140) #Use the labels derived from the sentiment_counts variable
        axes[1, 0].set_title('Sentiment Pie Chart')
    
        # Confusion Matrix
        sns.heatmap(confusion_matrix(y_test, mnb_pred), annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
        axes[1, 1].set_title('Confusion Matrix')
        axes[1, 1].set_xlabel('Predicted')
        axes[1, 1].set_ylabel('Actual')
    
        plt.tight_layout()
        plt.show()

    create_dashboard()
    return


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Train Support Vector Machine (SVM) Model
    This trains a SVM with a maximum of 1000 iterations on the TF IDF features. It predicts test laels then prints accuracy and a detailed report.
    """
    )
    return


@app.cell
def _(
    LinearSVC,
    X_test_tfidf,
    X_train_tfidf,
    accuracy_score,
    classification_report,
    y_test,
    y_train,
):
    svm = LinearSVC(max_iter=1000)
    svm.fit(X_train_tfidf, y_train)

    svm_pred = svm.predict(X_test_tfidf)

    print("SVM Accuracy:", accuracy_score(y_test, svm_pred))
    print("\nSVM Classification Report:\n", classification_report(y_test, svm_pred))
    return (svm,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Train Logistic Regression model
    This trains a Logistic Regression model with up to 100 iterations on the TF IDF features. Predicts labels for the test data and prints the accuracy and detailed report.
    """
    )
    return


@app.cell
def _(
    LogisticRegression,
    X_test_tfidf,
    X_train_tfidf,
    accuracy_score,
    classification_report,
    y_test,
    y_train,
):
    logreg = LogisticRegression(max_iter=100)
    logreg.fit(X_train_tfidf, y_train)

    logreg_pred = logreg.predict(X_test_tfidf)

    print("Logistic Regression Accuracy:", accuracy_score(y_test, logreg_pred))
    print("\nLogistic Regression Classification Report:\n", classification_report(y_test, logreg_pred))
    return (logreg,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Make predictions on sample tweets
    Take sample tweets and transforms them into TF IDF features using same vectoriser. Predicts their sentiment using the trained BernoulliNB, SVM and Logistic Regression models and prints the results for each classifier. 1 stands for positive and 0 for negative.
    """
    )
    return


@app.cell
def _(bnb, logreg, svm, vectorizer):
    sample_tweets = ["I love this!", "I hate that!", "It was okay, not great."]
    sample_vec = vectorizer.transform(sample_tweets)

    print("\nSample Predictions:")
    print("BernoulliNB:", bnb.predict(sample_vec))
    print("SVM:", svm.predict(sample_vec))
    print("Logistic Regression:", logreg.predict(sample_vec))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
