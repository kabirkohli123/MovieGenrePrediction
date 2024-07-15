import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from tqdm import tqdm
import csv
import sqlite3
import itertools

from sklearn.decomposition import PCA


genre_list = [ 'action', 'adult', 'adventure', 'animation', 'biography','comedy', 'crime','documentary','family', 'fantasy', 'game-show', 'history', 'horror', 'music','musical','mystery','news','reality-tv','romance','sci-fi','short','sport','talk-show','thriller','war','western']

fallback_genre = 'Unknown'

try:
    with tqdm(total=50, desc="Loading Train Data") as pbar:
        train_data = pd.read_csv('test_data.txt', sep=':::', header=None, names=['SerialNumber', 'MOVIE_NAME','GENRE','MOVIE_PLOT'],engine='python')
        pbar.update(50)

except Exception as e:
    print(f"Error loading train_data: {e}")
    raise


X_train = train_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())
genre_labels = [genre.split(', ') for genre in train_data['GENRE']]
mlb = MultiLabelBinarizer()
y_train = mlb.fit_transform(genre_labels)


Tfidf_vectorizer = TfidfVectorizer(max_features=5000)

with tqdm(total=50, desc="Vectorization Training Data") as pbar:
    X_train = Tfidf_vectorizer.fit_transform(X_train)
    pbar.update(50)

with tqdm(total=50,desc="Training Model") as pbar:
    naive_bayes = MultinomialNB()
    multi_output_classifier = MultiOutputClassifier(naive_bayes)  
    multi_output_classifier.fit(X_train, y_train)
    pbar.update(50)

try:
    with tqdm(total=50, desc="Loading Test Data") as pbar:
        test_data = pd.read_csv('test_data.txt', sep=':::', header=None, names
                                =['SerialNumber', 'MOVIE_NAME','GENRE','MOVIE_PLOT'],engine='python')

except Exception as e:
    print(f"Error loading test_data: {e}")      
    raise

X_test = test_data['MOVIE_PLOT'].astype(str).apply(lambda doc: doc.lower())

with tqdm(total=50, desc="Vectorization Test Data") as pbar:
    X_test_tdidf = Tfidf_vectorizer.transform(X_test)
    pbar.update(50)

with tqdm(total=50,desc="Predicting on Test DAta") as pbar:
    y_pred = multi_output_classifier.predict(X_test_tdidf)  
    pbar.update(50)

test_movie_names = test_data['MOVIE_NAME']
predicted_genres = mlb.inverse_transform(y_pred)
test_results = pd.DataFrame({'MOVIE_NAME':test_movie_names, 'PREDICTED_GENRES': predicted_genres})   
test_results['PREDICTED_GENRES'] = test_results['PREDICTED_GENRES'].apply(lambda genres: [fallback_genre] if len(genres) == 0 else genres)

with open("model_evaluation.txt","w", encoding="utf-8") as output_file:
    for _, row in test_results.iterrows():
         movie_name = row['MOVIE_NAME']
         genre_str = ', '.join(row['PREDICTED_GENRES'])
         output_file.write(f"{movie_name} ::: {genre_str}\n")
   

y_train_pred = multi_output_classifier.predict(X_test_tdidf)

accuracy = accuracy_score(y_train, y_train_pred)
precision = precision_score(y_train,y_train_pred,average='micro')
recall = recall_score(y_train,y_train_pred,average='micro')
f1 = f1_score(y_train,y_train_pred,average='micro')

with open("model_evaluation.txt","a", encoding="utf-8") as output_file:
    output_file.write("\nModel Evaluation Metrices:\n")
    output_file.write(f"Accuracy: {accuracy *100:.2f}\n")
    output_file.write(f"Precision: {precision:.2f}\n")
    output_file.write(f"Recall: {recall:.2f}\n")
    output_file.write(f"F1 Score: {f1:.2f}\n")


print("Model evaluation results and metrics have been saved to 'model_evaluation.txt'.")









