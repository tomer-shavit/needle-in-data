import pandas as pd
import numpy as np


def a1():
    df = pd.read_csv('../music_festivals.csv')

    target_words = ['annual', 'music', 'festival', 'soul', 'jazz', 'belgium', 'hungary', 'israel', 'rock', 'dance',
                    'desert', 'electronic', 'arts']

    df['Description'] = df['Description'].str.lower().str.replace('[^\w\s]', '')

    def term_frequency(term, document):
        words = document.split()
        term_count = words.count(term)
        return term_count / len(words)

    def document_frequency(term, documents):
        doc_count = sum(1 for doc in documents if term in doc.split())
        return doc_count

    def tfidf(df, target_words):
        N = len(df)
        tfidf_matrix = pd.DataFrame(index=df.index, columns=target_words)
        descriptions = df['Description'].tolist()

        for word in target_words:
            df_word = document_frequency(word, descriptions)
            idf_word = np.log(N / (1 + df_word))

            for i, desc in enumerate(descriptions):
                tf_word = term_frequency(word, desc)
                tfidf_matrix.loc[i, word] = tf_word * idf_word

        return tfidf_matrix

    tfidf_values = tfidf(df, target_words)

    tfidf_values.insert(0, 'Music Festival', df['Music Festival'])

    tfidf_values.to_csv('../output/tfidf_values.csv', index=False)

    print(tfidf_values)


if __name__ == "__main__":
    # a1()
    pass
