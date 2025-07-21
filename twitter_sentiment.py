import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import LinearSVC
    from sklearn.metrics import accuracy_score, classification_report
    return TfidfVectorizer, mo, pd, train_test_split


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
def _(pd):
    df = pd.read_csv('data/training.1600000.processed.noemoticon.csv', encoding='latin-1', header=None)
    df = df[[0,5]]
    df.columns = ['polarity', 'text']
    print(df.head())
    return (df,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Keep positive and negative sentiments
    Remove the neutral tweets where polarity is 2. Then map labels so that 0 represents negative and then assign 1 for positive. Then calculate the number of positive and negative left in the dataset.
    """
    )
    return


@app.cell
def _(df):
    def sentiment(df):
        df = df[df.polarity !=2]
        df['polarity'] = df['polarity'].map({0: 0, 4: 1})
        return df

    sentiment(df)
    print(df['polarity'].value_counts())
    print(df.head())
    return (sentiment,)


@app.cell
def _(mo):
    mo.md(
        r"""
    ## Now clean the tweets
    Simple function to convert all text to lowercase so it is consistent then show original and cleaned tweets
    """
    )
    return


@app.cell
def _(df, sentiment):
    def clean_text(text):
        return text.lower()

    df_1 = sentiment(df)
    df_1['clean_text'] = df_1['text'].apply(clean_text)

    print(df_1[['text', 'clean_text']].head())
    return (df_1,)


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
def _(df_1, train_test_split):
    X_train, X_test, y_train, y_test = train_test_split(
        df_1['clean_text'],
        df_1['polarity'],
        test_size=0.2,
        random_state=42
    )

    print("Train size:", len(X_train))
    print("Test size:", len(X_test))
    return X_test, X_train


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
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
