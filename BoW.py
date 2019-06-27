from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
import math
import collections
import pandas

"""open and close the files"""
fileOpenReal = open("clean_real-Train.txt", "r")
realTrain = fileOpenReal.read()
fileOpenReal.close()
fileOpenFake = open("clean_fake-Train.txt", "r")
fakeTrain = fileOpenFake.read()
fileOpenFake.close()

# create a variable as totalTrain that keep total train dataset
totalTrain = realTrain + fakeTrain
# token_pattern using for one character
vecTotal = CountVectorizer(token_pattern=r'\b\w+\b')
total = vecTotal.fit_transform([totalTrain])
cardinality = len(total.toarray()[0])

"""unigram for real train data"""
vecReal = CountVectorizer(token_pattern=r'\b\w+\b')
real = vecReal.fit_transform([realTrain])
realUniq = vecReal.get_feature_names()
realUniqCount = real.toarray()
dicReal = dict(zip(realUniq, realUniqCount[0]))
realCount = sum(realUniqCount[0])

"""unigram for fake train data"""
vecFake = CountVectorizer(token_pattern=r'\b\w+\b')
fake = vecFake.fit_transform([fakeTrain])
fakeUniq = vecFake.get_feature_names()
fakeUniqCount = fake.toarray()
dicFake = dict(zip(fakeUniq, fakeUniqCount[0]))
fakeCount = sum(fakeUniqCount[0])

"""find unigram probability"""
dicReal_prob = {}
dicFake_prob = {}
for key, value in dicReal.items():
    dicReal_prob[key] = math.log(((value + 1) / (realCount + cardinality)))

for key, value in dicFake.items():
    dicFake_prob[key] = math.log(((value + 1) / (fakeCount + cardinality)))


vecBiTotal = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=1)
totalB = vecBiTotal.fit_transform([totalTrain])
cardinalityB = len(totalB.toarray()[0])

"""bigram for real train data"""
vecBReal = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=1)
b_Real = vecBReal.fit_transform([realTrain])
b_RealUniq = vecBReal.get_feature_names()
b_RealUniqCount = b_Real.toarray()
dicB_Real = dict(zip(b_RealUniq, b_RealUniqCount[0]))
b_RealCount = sum(b_RealUniqCount[0])

"""bigram for fake train data"""
vecBFake = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=1)
b_Fake = vecBFake.fit_transform([fakeTrain])
b_FakeUniq = vecBFake.get_feature_names()
b_FakeUniqCount = b_Fake.toarray()
dicB_Fake = dict(zip(b_FakeUniq, b_FakeUniqCount[0]))
b_FakeCount = sum(b_FakeUniqCount[0])

"""find bigram probability"""
b_Real_Prob = {}
for key, value in dicB_Real.items():
    firstword = key.split()[0]
    b_Real_Prob[key] = math.log((value + 1)/(dicReal[firstword] + cardinalityB))

b_Fake_Prob = {}
for key, value in dicB_Fake.items():
    firstword = key.split()[0]
    b_Fake_Prob[key] = math.log((value + 1)/(dicFake[firstword] + cardinalityB))

realLineCount = len(realTrain.split('\n'))
fakeLineCount = len(fakeTrain.split('\n'))
realClass = math.log((realLineCount / (realLineCount + fakeLineCount)))
fakeClass = math.log((fakeLineCount / (realLineCount + fakeLineCount)))

"""Read test data"""
testData = pandas.read_csv('test.csv', sep=',')
count = 0
"""accuracy for bigram"""
for row in testData.itertuples():
    prob_real = realClass
    prob_fake = fakeClass
    #print(row._2)
    vecTestSentence = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b', min_df=1)
    b_TestSentence = vecTestSentence.fit_transform([row.Id])
    b_TestSentenceU = vecTestSentence.get_feature_names()

    for j in range(len(b_TestSentenceU)):
        if b_TestSentenceU[j] in dicB_Real:
            prob_real += dicB_Real[b_TestSentenceU[j]]
        else:
            prob_real += math.log(1/cardinalityB)
        if b_TestSentenceU[j] in dicB_Fake:
            prob_fake += dicB_Fake[b_TestSentenceU[j]]
        else:
            prob_fake += math.log(1/cardinalityB)
    if prob_real > prob_fake:
        if row._2 == "real":
            count += 1
    elif prob_fake > prob_real:
        if row._2 == "fake":
            count += 1

accuracy = 100*(count / len(testData))
print(accuracy)

""" TF-IDF"""
vectReal = TfidfVectorizer(sublinear_tf=True, analyzer='word', ngram_range=(1, 1))
tfidfReal = vectReal.fit_transform([realTrain])
tfidfRealList = dict(zip(vectReal.get_feature_names(), tfidfReal.toarray()[0]))

vectFake = TfidfVectorizer(sublinear_tf=True, analyzer='word', ngram_range=(1, 1))
tfidfFake = vectFake.fit_transform([fakeTrain])
tfidfFakeList = dict(zip(vectFake.get_feature_names(), tfidfFake.toarray()[0]))


"""TF-IDF stopwords"""
vectRealStop = TfidfVectorizer(sublinear_tf=True, analyzer='word', stop_words=ENGLISH_STOP_WORDS, ngram_range=(1, 1))
tfidfRealStop = vectRealStop.fit_transform([realTrain])
tfidfRealListStop = dict(zip(vectRealStop.get_feature_names(), tfidfRealStop.toarray()[0]))

vectFakeStop = TfidfVectorizer(sublinear_tf=True, analyzer='word', stop_words= ENGLISH_STOP_WORDS, ngram_range=(1, 1))
tfidfFakeStop = vectFakeStop.fit_transform([fakeTrain])
tfidfFakeListStop = dict(zip(vectFakeStop.get_feature_names(), tfidfFakeStop.toarray()[0]))
