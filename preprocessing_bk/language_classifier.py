import pandas as pd
import numpy as np

from itertools import permutations
import collections

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

'''
IDEA : raw_data(books.csv)의 book_title은 해당 언어로 쓰여져 있기 때문에 book_title을 분석해서 language 결측치 처리
PROCESS : feature 추출 -> PCA -> Random Forest Classifier (about 99.6% acc)
OUTPUT : new_books
'''

books = pd.read_csv('../data/raw/books.csv')
tmp_books = books[books.language.notnull()]

def make_features(tmp_books) : # trainset + testset

    # Define a list of commonly found punctuations
    punc = ('!', "," ,"\'" ,";" ,"\"", ".", "-" ,"?")
    vowels=['a','e','i','o','u']
    # Define a list of double consecutive vowels which are typically found in Dutch and Afrikaans languages
    same_consecutive_vowels = ['aa','ee', 'ii', 'oo', 'uu'] 
    consecutive_vowels = [''.join(p) for p in permutations(vowels,2)]
    dutch_combos = ['ij']

    # Create a pre-defined set of features based on the "text" column in order to allow us to characterize the string
    tmp_books['word_count'] = tmp_books['book_title'].apply(lambda x : len(x.split()))
    tmp_books['character_count'] = tmp_books['book_title'].apply(lambda x : len(x.replace(" ","")))
    tmp_books['word_density'] = tmp_books['word_count'] / (tmp_books['character_count'] + 1)
    tmp_books['punc_count'] = tmp_books['book_title'].apply(lambda x : len([a for a in x if a in punc]))
    tmp_books['v_char_count'] = tmp_books['book_title'].apply(lambda x : len([a for a in x if a.casefold() == 'v']))
    tmp_books['w_char_count'] = tmp_books['book_title'].apply(lambda x : len([a for a in x if a.casefold() == 'w']))
    tmp_books['ij_char_count'] = tmp_books['book_title'].apply(lambda x : sum([any(d_c in a for d_c in dutch_combos) for a in x.split()]))
    tmp_books['num_double_consec_vowels'] = tmp_books['book_title'].apply(lambda x : sum([any(c_v in a for c_v in same_consecutive_vowels) for a in x.split()]))
    tmp_books['num_consec_vowels'] = tmp_books['book_title'].apply(lambda x : sum([any(c_v in a for c_v in consecutive_vowels) for a in x.split()]))
    tmp_books['num_vowels'] = tmp_books['book_title'].apply(lambda x : sum([any(v in a for v in vowels) for a in x.split()]))
    tmp_books['vowel_density'] = tmp_books['num_vowels']/tmp_books['word_count']
    tmp_books['capitals'] = tmp_books['book_title'].apply(lambda comment: sum(1 for c in comment if c.isupper()))
    tmp_books['caps_vs_length'] = tmp_books.apply(lambda row: float(row['capitals'])/float(row['character_count']),axis=1)
    tmp_books['num_exclamation_marks'] = tmp_books['book_title'].apply(lambda x: x.count('!'))
    tmp_books['num_question_marks'] = tmp_books['book_title'].apply(lambda x: x.count('?'))
    tmp_books['num_punctuation'] = tmp_books['book_title'].apply(lambda x: sum(x.count(w) for w in punc))
    tmp_books['num_unique_words'] = tmp_books['book_title'].apply(lambda x: len(set(w for w in x.split())))
    tmp_books['num_repeated_words'] = tmp_books['book_title'].apply(lambda x: len([w for w in collections.Counter(x.split()).values() if w > 1]))
    tmp_books['words_vs_unique'] = tmp_books['num_unique_words'] / tmp_books['word_count']
    tmp_books['encode_ascii'] = np.nan
    
    for i in range(len(tmp_books)):
        try:
            tmp_books['book_title'].iloc[i].encode(encoding='utf-8').decode('ascii')
        except UnicodeDecodeError:
            tmp_books['encode_ascii'].iloc[i] = 0
        else:
            tmp_books['encode_ascii'].iloc[i] = 1
    
    return tmp_books

def train_test_split(tmp_books) : 
    
    test = tmp_books[tmp_books.language.isnull()]
    train = tmp_books[tmp_books.language.notnull()]
    
    feature_cols = tmp_books.columns[10:]
    
    X_test = test[feature_cols]
    y_test = test['language']
    
    X_train = train[feature_cols]
    y_train = train['language']

    # from sklearn.model_selection import train_test_split
    
    # X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size = 0.2)
    
    data = {
        'X_train' : X_train,
        'X_test' : X_test,
        'y_train' : y_train,
        'y_valid' : y_test
    }
    
    return data

def fit_scaling_n_pca(data) : #trainset only
    
    # Standardize the data
    scaler = StandardScaler()
    # Fit on training set only.
    scaler.fit(data['X_train'])

    # Make an instance of the model to retain 95% of the variance within the old features.
    pca = PCA(.95)
    pca.fit(data['X_train'])
    
    return scaler, pca

def transform(data, scaler, pca) : # train + test set 
    data['X_train'] = scaler.transform(data['X_train'])
    data['X_train'] = pca.transform(data['X_train'])
    data['X_test'] = scaler.transform(data['X_test'])
    data['X_test'] = pca.transform(data['X_test'])
    
    return data

def train_RF(data, train = True) : 
    
    if train : 
        from sklearn.ensemble import RandomForestClassifier
        
        rf_clf = RandomForestClassifier(n_estimators=100) # Create Random Forest classifer object
        rf_clf = rf_clf.fit(data['X_train'],data['y_train']) # Fit/Train Random Forest Classifer on training set
    else : 
        pass
    
    return rf_clf

def test_RF(rf_clf, data, train = False) : 
    
    y_pred = rf_clf.predict(data['X_test']) #Predict the response for test dataset
    
    return y_pred

def fill_lang(df, y_pred) : 
    df.loc[df.language.isnull(), 'language'] = y_pred
    
    return df


new_books = books.copy()
new_books = make_features(new_books)
data = train_test_split(new_books)
scaler, pca = fit_scaling_n_pca(data)
data = transform(data, scaler, pca)
rf_clf = train_RF(data, train = True)
y_pred = test_RF(rf_clf, data)
new_books = fill_lang(new_books, y_pred)
new_books = new_books.iloc[:,:10]

# new_books.to_csv('../data/preprocessed/books_lang.csv', index = False)