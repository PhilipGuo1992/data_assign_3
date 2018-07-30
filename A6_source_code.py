import sklearn
import pandas
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd
import csv
from collections import Counter
import random
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score


def some(x, n):
    return x.ix[random.sample(x.index, n)]

def class_number(class_string):
    if class_string == 'bg':
        return 1
    if class_string == 'mk':
        return 2
    if class_string == 'bs':
        return 3
    if class_string == 'hr':
        return 4
    if class_string == 'sr':
        return 5
    if class_string == 'cz':
        return 6
    if class_string == 'sk':
        return 7
    if class_string == 'es-AR':
        return 8
    if class_string == 'es-ES':
        return 9
    if class_string == 'pt-BR':
        return 10
    if class_string == 'pt-PT':
        return 11
    if class_string == 'id':
        return 12
    if class_string == 'my':
        return 13
    
# bigram_vectorizer = CountVectorizer(ngram_range=(1, 2), token_pattern=r'\b\w+\b', min_df=1)
# analyzier = vectorizer.build_analyzer()
# transformer = TfidfTransformer(smooth_idf=False)


df = pd.read_csv('train.txt', sep='\t', header=None, names=['sample', 'target'])
df = df[df.target != 'xx']
df['class'] = df['target'].map(class_number)


feature = np.array(df)[:, 0]

label = np.array(df)[:, 1]
features = ['Counter', 'Index']
df2 = pd.DataFrame
df2 = df.loc[df['class'] == 1].sample(n=300).\
    append(df.loc[df['class'] == 2].sample(n=300), ignore_index=True).\
    append(df.loc[df['class'] == 3].sample(n=300), ignore_index=True).\
    append(df.loc[df['class'] == 4].sample(n=300), ignore_index=True).\
    append(df.loc[df['class'] == 5].sample(n=300), ignore_index=True).\
    append(df.loc[df['class'] == 6].sample(n=300), ignore_index=True).\
    append(df.loc[df['class'] == 7].sample(n=300), ignore_index=True).\
    append(df.loc[df['class'] == 8].sample(n=300), ignore_index=True).\
    append(df.loc[df['class'] == 9].sample(n=300), ignore_index=True).\
    append(df.loc[df['class'] == 10].sample(n=300), ignore_index=True).\
    append(df.loc[df['class'] == 11].sample(n=300), ignore_index=True).\
    append(df.loc[df['class'] == 12].sample(n=300), ignore_index=True).\
    append(df.loc[df['class'] == 13].sample(n=300), ignore_index=True)

vectorizer = TfidfVectorizer()
x = vectorizer.fit_transform(df2['sample']).toarray()
y = df2['class'].values

#PCA plot
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data=principalComponents, columns=['principal component 1', 'principal component 2'])
finalDf = pd.concat([principalDf, df2[['target']]], axis=1)
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(1, 1, 1)
ax.set_xlabel('principal component 1', fontsize = 10)
ax.set_ylabel('principal component 2', fontsize = 10)
targets = ['bg', 'mk', 'bs', 'hr', 'sr', 'cz', 'sk', 'es_AR', 'es_ES', 'pt-BR', 'pt-PT', 'id', 'my']
colors = ['black', '#00FFFF', '#F5F5DC', '#FFEBCD', '#0000FF', '#A52A2A', '#DEB887', '#D2691E', '#DC143C', '#00008B', '#006400', '#FF8C00', '#8B0000']
for target, color in zip(targets, colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1'],
               finalDf.loc[indicesToKeep, 'principal component 2'],
               c=color,
               s=10)

ax.legend(targets)
ax.grid()
plt.show()
# sc = StandardScaler()

X = df['sample']
Y = df['class']
x_train, y_train = X, Y
df3 = pd.read_csv('test-gold.txt', sep='\t', header=None, names=['sample', 'target'])
df3['class'] = df3['target'].map(class_number)
df3 = df3[df3.target != 'xx']
x_test, y_test = df3['sample'], np.array(df3)[:, 2]

print("Logistic Regression")
for k in range(30, 60):
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('feat', SelectKBest(chi2, k=k)),
        ('clf', LogisticRegression())
    ])
    pipeline.fit(x_train, y_train)
    pipe_result = pipeline.predict(x_test)
    #print(pipe_result)
    print("K = ", k)
    print(r2_score(y_test, pipe_result))

print("NB")
for k in range(30, 60):
    pipeline2 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('feat', SelectKBest(chi2, k=k)),
        ('clf', BernoulliNB())
    ])
    pipeline2.fit(x_train, y_train)
    pipe_result = pipeline2.predict(x_test)
    # print(pipe_result)
    print("K = ", k)
    print(r2_score(y_test, pipe_result))

print("Decision Tree")
for k in range(30, 60):
    pipeline3 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('feat', SelectKBest(chi2, k=k)),
        ('clf', DecisionTreeClassifier())
    ])
    pipeline3.fit(x_train, y_train)
    pipe_result = pipeline3.predict(x_test)
    # print(pipe_result)
    print("K = ", k)
    print(r2_score(y_test, pipe_result))

print("LinearSVC")
for k in range(30, 60):
    pipeline4 = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('feat', SelectKBest(chi2, k=k)),
        ('clf', LinearSVC())
    ])
    pipeline4.fit(x_train, y_train)
    pipe_result = pipeline4.predict(x_test)
    #print(pipe_result)
    print("K = ", k)
    print(r2_score(y_test, pipe_result))



#test code
# x_train = df['sample']
# x_train = TfidfVectorizer().fit_transform(x_train).toarray
# x_train = [x_train]
# y_train = np.array(df)[:, 2]
# clf = LinearRegression()
# selection = SelectKBest(chi2, k=10)
# selection.fit_transform(x_train, y_train)
# clf.fit(x_train, y_train)
# x_test = TfidfVectorizer().fit_transform(x_test).toarray
# result = clf.predict(x_test)
# print(result)
# pre_tes = np.mean(result == y_test)
# print(pre_tes)








