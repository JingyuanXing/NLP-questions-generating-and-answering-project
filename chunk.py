
# the function that connects together these three parts of IE Architecture
def ie_preprocess(document):
    # sentence segmenter
    sentences = nltk.sent_tokenize(document)
    # word tokenizer
    sentences = [nltk.word_tokenize(sent) for sent in sentences]
    # part-of-speech tagger
    sentences = [nltk.pos_tag(sent) for sent in sentences]

    return sentences


# NONE phrase chunking (divide into levels using tree structure)
def np_chunking(sentences):
    # the chunk grammar 
    # no.1: single regular-expression rule: <DT> followed by any number of <JJ> and <NN>
    # no.2: sequences of proper nouns
    # no.3: two or more consecutive nouns
    grammar = "NP: {<DT>?<JJ>*<NN>}{<NNP>+}{<NN>+}"
    cp = nltk.RegexpParser(grammar)
    result = cp.parse(sentence)

    return result

# for evaluation, use unigram, bigram, trigram

class UnigramChunker(nltk.ChunkParserI):
    def __init__(self, train_sents): # [1]
        train_data = [[(t,c) for w,t,c in nltk.chunk.tree2conlltags(sent)]
                      for sent in train_sents]
        self.tagger = nltk.UnigramTagger(train_data) # [2]

    def parse(self, sentence): # [3]
        pos_tags = [pos for (word,pos) in sentence]
        tagged_pos_tags = self.tagger.tag(pos_tags)
        chunktags = [chunktag for (pos, chunktag) in tagged_pos_tags]
        conlltags = [(word, pos, chunktag) for ((word,pos),chunktag)
                     in zip(sentence, chunktags)]
        return nltk.chunk.conlltags2tree(conlltags)

