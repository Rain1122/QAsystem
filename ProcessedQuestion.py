from nltk import pos_tag,word_tokenize,ne_chunk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import wordnet,stopwords

class ProcessedQuestion:
    def __init__(self, question, useStemmer = False, useSynonyms = False, removeStopwords = False):
        self.question = question
        self.useStemmer = useStemmer
        self.useSynonyms = useSynonyms
        self.removeStopwords = removeStopwords
        self.stopWords = stopwords.words("english")
        self.stem = lambda k : k.lower()
        if self.useStemmer:
            ps = PorterStemmer()
            self.stem = ps.stem



        self.qType = self.determineQuestionType(question)
        self.searchQuery = self.buildSearchQuery(question)
        self.qVector = self.getQueryVector(self.searchQuery)

    def determineQuestionType(self, question):
        questionTaggers = ['WP','WDT','WP$','WRB']
        qPOS = pos_tag(word_tokenize(question))
        qTags = []
        for token in qPOS:
            if token[1] in questionTaggers:
                qTags.append(token[1])
        qType = ''
        if(len(qTags)>1):
            qType = 'complex'
        elif(len(qTags) == 1):
            qType = qTags[0]
        else:
            qType = "None"
        return qType

    def buildSearchQuery(self, question):
        qPOS = pos_tag(word_tokenize(question))
        searchQuery = []
        questionTaggers = ['WP','WDT','WP$','WRB']
        for tag in qPOS:
            if tag[1] in questionTaggers:
                continue
            else:
                searchQuery.append(tag[0])
                if(self.useSynonyms):
                    syn = self.getSynonyms(tag[0])
                    if(len(syn) > 0):
                        searchQuery.extend(syn)
        return searchQuery

    def getQueryVector(self, searchQuery):
        vector = {}
        for token in searchQuery:
            if self.removeStopwords:
                if token in self.stopWords:
                    continue
            token = self.stem(token)
            if token in vector.keys():
                vector[token] += 1
            else:
                vector[token] = 1
        return vector

    def getSynonyms(word):
        synonyms = []
        for syn in wordnet.synsets(word):
            for l in syn.lemmas():
                w = l.name().lower()
                synonyms.extend(w.split("_"))
        return list(set(synonyms))

    def __repr__(self):
        msg = "Q: " + self.question + "\n"
        msg += "QType: " + self.qType + "\n"
        msg += "QVector: " + str(self.qVector) + "\n"
        return msg