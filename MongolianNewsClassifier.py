# Loading all the required libraries
import pandas as pd
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Library for regular expretion
import re
import string

data = pd.read_csv("news.csv",encoding='UTF-8')
data = data[['CATEGORY','NEWS']]


# Printing all different types of categories
data.CATEGORY.unique()
data.groupby('CATEGORY').describe()

# Файлаас зогсох үг буюу туслах үгнүүдийг унших
with open('stopwords.txt', 'r', encoding="utf8") as f:
    stopwords = f.read().split('\n')
# Файлаас залгаваруудыг унших
with open('rules.txt', 'r', encoding="utf8") as f:
    rules = f.read().split('\n')
# Залгавар үгнүүдийн REGEX үүсгэх 
rulesREGEXP = '$|'.join(rules)+'$'
    
def parse(text):
    text_sentences = text.split(' ')
    tokens = text.split(' ')
    # том үсгүүдийг болиулах
    tokens = [w.lower() for w in tokens]
    # үг бүрээс тэмдэгтүүдийг хасах
    table = str.maketrans('', '', string.punctuation)
    stripped = [w.translate(table) for w in tokens]
    # текст бус үгүүдийг хасах
    words = [word for word in tokens if word.isalpha()]
    # stopword уудыг хасах
    words = [w for w in words if not w in stopwords]
    # stemming
    words = [re.sub(rulesREGEXP, '', w) for w in words if len(w) >= 6]
    words = ' '.join(words)
    return words

for i in range(0, len(data)):
    data['NEWS'][i] = parse(data['NEWS'][i])

# Converting category column into numeric target NUM_CATEGORY column
data['NUM_CATEGORY']= data.CATEGORY.map({'байгал орчин':0,'боловсрол':1,'спорт':2,
    'технологи':3,
    'улс төр':4,
    'урлаг соёл':5,
    'хууль':6,
    'эдийн засаг':7,
    'эрүүл мэнд':8})
data.head()
data = data[data['NEWS'].notnull()]

# Splitting dataset into 80% training set and 20% test set
x_train, x_test, y_train, y_test = train_test_split(data.NEWS, data.NUM_CATEGORY, test_size = 0.1, random_state=0)

# Here we convert our dataset into a Bag Of Word model using a Bigram model
vect = CountVectorizer(ngram_range=(2,2))
# Converting traning features into numeric vector
X_train = vect.fit_transform(x_train)
xTrain = np.array(X_train)
# Converting training labels into numeric vector
X_test = vect.transform(x_test)

# Training and Predicting the data
mnb = MultinomialNB(alpha =0.2)
mnb.fit(X_train,y_train)
result= mnb.predict(X_test)

# Printing accuracy of the our model
accuracy_score(result,y_test)

# This function return the class of the input news
def predict_news(news):
    test = vect.transform(news)
    pred = mnb.predict(test)
    if pred  == 0:
         return 'байгал орчин'
    elif pred == 1:
        return 'боловсрол'
    elif pred == 2:
        return 'спорт'
    elif pred == 3:
        return 'технологи'
    elif pred == 4:
        return 'улс төр'
    elif pred == 5:
        return 'урлаг соёл'
    elif pred == 6:
        return 'хууль'
    elif pred == 7:
        return 'эдийн засаг'
    else:
        return 'эрүүл мэнд'
    
# Copy and paste the news headline in 'x'
x = "Нэгдсэн дүнг энэ сарын 25-ны өдөр гаргана  гэж Нийслэлийн ЗДТГ-ын Хэвлэл мэдээлэл, олон нийттэй харилцах хэлтсээс мэдээллээ"
y = []
y.append(parse(x))
r = predict_news(y)
print (r)

# Printing the confusion matrix of our prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, result)
cm