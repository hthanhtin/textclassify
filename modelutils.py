import os, glob, re
import emoji
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

class modelUtils:
    def __init__(self):
        self.stopwordsPath=os.path.join(os.getcwd(),"data","vi_stopwords.txt")

    def file2str(self, txtfilename):
        f=open(txtfilename,"r", encoding="utf8")
        str=f.read().strip()
        str=str.lower()
        #xoá emoji
        str=emoji.demojize(str)
        #xoá kí tự đặc biệt
        str=re.sub(r'\d', ' ', str)
        str = re.sub(r'[!@#$%^&*()_\-+/,\.:;~]', ' ', str)
        x=[w.strip() for w in str.split() if len(w.strip())>1 and len(w.strip())<8]
        f.close()
        return " ".join(x)
    def fixstring(self, str):
        str = str.lower()
        # xoá emoji
        stw=self.getStopword()
        str = emoji.demojize(str)
        # xoá kí tự đặc biệt
        str = re.sub(r'\d', ' ', str)
        str = re.sub(r'[!@#$%^&*()_\-+/,\.:;~]', ' ', str)
        x = [w.strip() for w in str.split() if len(w.strip()) > 1 and len(w.strip()) < 8 and w not in stw]
        return " ".join(x)
    def getStopword(self):
        X=[w.strip() for w in open(self.stopwordsPath,"r", encoding="utf8").readlines()]
        return X
    def getTexts_Labels(self, Path):
        Texts=[]
        Labels=[]
        posPath=os.path.join(Path,"pos")
        for f in glob.glob(posPath+"/*"):
            str=self.file2str(f)
            Texts.append(str)
            Labels.append(1)
        negPath = os.path.join(Path, "neg")
        for f in glob.glob(negPath+"/*"):
            str=self.file2str(f)
            Texts.append(str)
            Labels.append(0)
        return Texts, Labels

class BayesModel:
    def __init__(self):
        self.vectorizer=None
        self.NBmodel=None
        self.X_train=None
        self.X_test=None
        self.y_train=None
        self.y_test=None
    def setData(self, Texts=None,Labels=None,TextPresent="Bow", ngram=1, stopwords=None, testsize=0.3, kernel="MultinomialNB"):
        self.Texts=Texts
        self.Labels=Labels
        self.TextPresent=TextPresent
        self.ngram=ngram
        self.stopwords=stopwords
        self.testsize=testsize
        self.kernel=kernel

        if TextPresent=="Bow":
            self.vectorizer = CountVectorizer(ngram_range=(ngram, ngram), stop_words=stopwords)
        else:
            self.vectorizer = TfidfVectorizer(ngram_range=(ngram, ngram), stop_words=stopwords)
        if Texts and Labels:
            self.X_train, self.X_test, self.y_train, self.y_test=train_test_split(self.Texts, self.Labels, test_size=testsize, random_state=5)
            self.X_train = self.vectorizer.fit_transform(self.X_train)
            self.X_train = np.array(self.X_train.toarray())
            self.X_test = self.vectorizer.transform(self.X_test)
            self.X_test = np.array(self.X_test.toarray())
    def train_model(self):
        x,y=None, None
        if self.kernel=="MultinomialNB":
            self.NBmodel=MultinomialNB()
            self.NBmodel.fit(self.X_train, self.y_train)
            x=self.NBmodel.score(self.X_train, self.y_train)
            y=self.NBmodel.score(self.X_test, self.y_test)
        elif self.kernel=="GaussianNB":
            self.NBmodel = GaussianNB()
            self.NBmodel.fit(self.X_train, self.y_train)
            x = self.NBmodel.score(self.X_train, self.y_train)
            y = self.NBmodel.score(self.X_test, self.y_test)
        else:
            self.NBmodel = BernoulliNB()
            self.NBmodel.fit(self.X_train, self.y_train)
            x = self.NBmodel.score(self.X_train, self.y_train)
            y = self.NBmodel.score(self.X_test, self.y_test)
        return x, y

    def save_model(self):
        import pickle
        filename=".".join([self.TextPresent, str(self.ngram),self.kernel])
        modelfilename=os.path.join(os.getcwd(),"models",filename + ".pkl")
        #print (modelfilename)
        with open(modelfilename, 'wb') as model_file:
            pickle.dump(self.NBmodel, model_file)
        #Vectorizer
        filename = ".".join([self.TextPresent, str(self.ngram), self.kernel,"vector"])
        vectorfilename = os.path.join(os.getcwd(), "models", filename + ".pkl")
        with open(vectorfilename, 'wb') as model_file:
            pickle.dump(self.vectorizer, model_file)

    def Str2Vector(self, str=None):
        vt=self.vectorizer.transform([str])
        vt=np.array(vt.toarray())
        return vt
    def Text2Class(self, str):
        m=modelUtils()
        str=m.fixstring(str)
        vt=self.Str2Vector(str)
        c=self.NBmodel.predict(vt)
        return c
    def loadModel(self):
        import pickle
        mn=".".join([self.TextPresent, str(self.ngram),self.kernel])
        vn=".".join([self.TextPresent, str(self.ngram),self.kernel,"vector"])
        modelfilename = os.path.join(os.getcwd(), "models", mn + ".pkl")
        vectorfilename = os.path.join(os.getcwd(), "models", vn + ".pkl")
        with open(modelfilename, "rb") as modelfile:
            self.NBmodel=pickle.load(modelfile)
        with open(vectorfilename,"rb") as vectorfile:
            self.vectorizer=pickle.load(vectorfile)
