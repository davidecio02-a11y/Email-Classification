# -*- coding: utf-8 -*-
"""
Created on Tue Jan 27 23:01:51 2026

@author: david
"""


import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_predict
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import VotingClassifier, StackingClassifier

import pickle


svc = SVC(kernel='linear', C=1, probability=True)
rfc = RandomForestClassifier(n_estimators=50, random_state=2)
lr = LogisticRegression(solver='liblinear', penalty='l2')
nmb = MultinomialNB()
# Dizionario degli algoritmi ML
algoritmi_ml = {
    "svc": svc,
    "random_forest": rfc,
    "logistic_regression": lr,
    "naive_bayes": nmb,
}
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay,precision_score

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nltk.download('punkt')
nltk.download('stopwords')
path = 'C:/Users/david/OneDrive/Desktop/'
df = pd.read_csv(path + "spam.csv", encoding='latin-1')
df.info()
df.head()
df = df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df = df.rename(columns={"v1": "label", "v2": "text"})

def train_classifier(alg, X, y):
    y_pred = cross_val_predict(alg, X,y,cv=5)
    return y_pred
# Preprocess text data
df["text"] = df["text"].str.lower()
df["text"] = df["text"].apply(word_tokenize)
#stop_words = set(stopwords.words("english"))
#df["text"] = df["text"].apply(lambda x: [word for word in x if word not in stop_words])
df["text"] = df["text"].apply(lambda x: " ".join(x))

# Feature Extraction
vectorizer = CountVectorizer(ngram_range=(1, 2), max_features=3500)
X = vectorizer.fit_transform(df["text"])
y = df["label"]

# Train the Model
nomi_algoritmi = []
accuratezze = []
precisioni = []
for name, algoritmo_ml in algoritmi_ml.items():
    y_pred = train_classifier(algoritmo_ml, X, y)
    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, pos_label='spam')
    nomi_algoritmi.append(name)
    accuratezze.append(accuracy)
    precisioni.append(precision)
    cm = confusion_matrix(y, y_pred, labels=['ham','spam'])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
    disp.plot(cmap=plt.cm.Blues)
    plt.title(f'Matrice di confusione - {name}')
    plt.show()
    
    print('\nalgoritmo:\n', name)
    print(classification_report(y, y_pred, digits=4))

df_risultati = pd.DataFrame({
    'Algoritmi': nomi_algoritmi,
    'Accuratezza': accuratezze,
    'Precision': precisioni
})    
df_long = df_risultati.melt(id_vars='Algoritmi', var_name='Metrica', value_name='Punteggio')

g = sns.catplot(data=df_long, x='Algoritmi', y='Punteggio', hue='Metrica', kind='bar')
# Ruota le etichette sull'asse x
g.set_xticklabels(rotation=45, horizontalalignment='right')

#ensemble classification
voting = VotingClassifier(estimators = [('svm', svc), ('nb', nmb),('random_forest', rfc)], voting='soft', weights = [0.4, 0.2, 0.4])
y_pred = cross_val_predict(voting, X,y,cv=5)
cm = confusion_matrix(y, y_pred, labels=['ham','spam'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matrice di confusione - voting')
plt.show()
print('\nvoting:\n')
print(classification_report(y, y_pred, digits=4))
#stacking
estimators = [('svm', svc), ('nb', nmb),('Logistic_regression', lr)]
final_estimator = RandomForestClassifier()
sc = StackingClassifier(estimators=estimators, final_estimator=final_estimator)
y_pred = cross_val_predict(sc, X,y,cv=5)
cm = confusion_matrix(y, y_pred, labels=['ham','spam'])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['ham', 'spam'])
disp.plot(cmap=plt.cm.Blues)
plt.title('Matrice di confusione - stacking')
plt.show()
print('\nstacking:\n')
print(classification_report(y, y_pred, digits=4))
# --- FASE 2: PREPARAZIONE PER STREAMLIT ---
voting.fit(X, y)
pickle.dump(vectorizer, open('vectorizer.pkl', 'wb'))
pickle.dump(voting, open('model.pkl', 'wb'))


