
import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import StanfordNERTagger
#nltk.download("popular")

r = open(file = "Development_data/set1/a1.txt", mode = "r")
text = r.read()

text = text.replace("(", ", ")
text = text.replace(")", ", ")
text_lines = text.splitlines()

sent_tokens = sent_tokenize(text_lines[4])
sents_tagged = [nltk.pos_tag(word_tokenize(sent)) for sent in sent_tokens]
sents_tagged

################################# adding grammar ###############################

grammar = r"""

NP: {<PRP\$|DT>?<JJ.*>*(<NN.*>+<CC>?<NN.*>*)(<POS>?<IN>?<TO>?<DT>?<JJ>*<NN.*>+)+(<,>(<POS>?<IN>?<TO>?<DT>?<JJ>*<NN.*>+)+(<,>)?)}    
    {<PRP\$|DT>?<JJ.*>*(<NN.*>+<CC>?<NN.*>*)(<,>(<POS>?<IN>?<TO>?<DT>?<JJ>*<NN.*>+)+(<,>)?)+} 
    {<PRP\$|DT>?<JJ.*>*(<NN.*>+<CC>?<NN.*>*)(<IN>?<TO>?<POS>?<DT>?<JJ>*(<NN.*>+<CC>?<JJ.*>*<NN.*>*)+)*}
    {<NN.*>+}
    {<PRP\$><NN>}
    {<PRP>}
    {<NP>+<IN><TO><NP>+}
      
VP: #{<NP>?<PRP>?(<VB>|<VBZ>|<VBP>|<VBN>|<VBD>)+<NP>*<PP>*}

    {<RB>*<VB.*><PRP\$>?<NP|CLAUSE>}
    {<VBD|VBZ><JJ.*>*<RB.*>*<VBN><TO>?<IN>?<NP|CLAUSE>(<CC><IN|TO><NP>)*}
    
CLAUSE: {<NP><VP><PP>?}


AD: {<RB>*<VBN|VBD>?<IN><PRP\$>*<NP>}
    {<IN><TO><NP>}


CLAUSE: {<WDT|WP|WP$|WRB|CC>?<AD>*<NP><VP><AD>*}
        {<AD>*<NP><AD>*<VP><AD>*}
        {<NP><VP>}

        
        
"""
cp = nltk.RegexpParser(grammar, loop=1)
result = cp.parse(sents_tagged[0])
print("tagged clauses: ", result, "\n")

############################ extract relations #################################

import re
import string

def NP_Comma_Process(wordslst):
    w2sent = " ".join(wordslst)
    w2_parts = w2sent.partition(",")
    print("w2_parts: ", w2_parts, "\n")
    if len(w2_parts) == 1:
        return None
    elif len(w2_parts)> 5:
        return None
    else:
        print("w2_parts[0]: ", w2_parts[0], "\n")
        return [(w2_parts[0],),tuple(["is"]),(w2_parts[2],)]

relations = []
from nltk import Tree
import string
if_main_clause = True
for i,child in enumerate(result):
    if isinstance(child, Tree) and child.label()=="CLAUSE":
        NP = list()
        Actions = list()
        NP2 = list()
        for j, gchild in enumerate(child):
            if not isinstance(gchild,Tree) and gchild[1] in ['WDT','WP','WP$','WRB','CC'] and j == 0:
                if_main_clause = False
            elif isinstance(gchild, Tree) and gchild.label() == "NP":
                words, tags = zip(*gchild.leaves())
                NP.append(" ".join(words))
                
            elif isinstance(gchild, Tree) and gchild.label() == "VP":
                for k, ggchild in enumerate(gchild):
                    if not isinstance(ggchild, Tree) and ggchild[1] in ['VBZ', "VBD", "VBN", "IN" , "TO"]:
                        Actions.append(ggchild[0])
                    elif isinstance(ggchild, Tree) and ggchild.label() in ['CLAUSE', "NP"]:
                        words, tags = zip(*ggchild)
                        NP2.append(" ".join(words))
                        comma_results = NP_Comma_Process(words)
                        if comma_results is not None:
                            relations.append(comma_results)
        a = [tuple(NP), tuple(Actions), tuple(NP2)]
        relations.append(a)

print("relations: ", relations, "\n")

############################### stanford NER ###################################

jar = 'stanford-ner/stanford-ner.jar'
# model = 'stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'
model = 'stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz'
# model = 'stanford-ner/classifiers/wikigold.conll.ser.gz'
# model = 'stanford-ner/classifiers/sentiment.ser.gz'
st = StanfordNERTagger(model, jar, encoding='utf-8')

# Qestion Generation 1
# Substitute type questions

questions = []
for r in relations:
    # print(r)
    subj = r[0][0]
    obj = r[2][0]
    tokenized_subj = word_tokenize(subj)
    classified_subj = st.tag(tokenized_subj)
    print(classified_subj)
    tokenized_obj = word_tokenize(obj)
    classified_obj = st.tag(tokenized_obj)
    print(classified_obj)

    # compile the words in the subjects and objects into phrases.
    phrase = "".join(["".join([x+" " for x in w]) for w in r])

    # general question with 'what' replacement. 
    phrase_no_subj = "".join(["".join([x+" " for x in w]) for w in [r[1], r[2]]])
    phrase_no_obj = "".join(["".join([x+" " for x in w]) for w in [r[0], r[1]]])
    questions.append("What " + phrase_no_subj + "?")
    questions.append(phrase_no_obj + "what " + "?")

    # specific questions targeting the named entity
    for word_tuple in classified_subj + classified_obj:
        if (word_tuple[1] == 'PERSON'):
            questions.append(re.sub(r'\b%s\b' % word_tuple[0], "who", phrase, 1) + "?")
        if (word_tuple[1] == 'LOCATION'):
            questions.append(re.sub(r'\b%s\b' % word_tuple[0], "where", phrase, 1) + "?")   
        if (word_tuple[1] == 'ORGANIZATION'):
            questions.append(re.sub(r'\b%s\b' % word_tuple[0], "what organization", phrase, 1) + "?")   

print("questions: ", questions, "\n")

################ polish questions by moving Wh to front ########################

def moveWh(input_sent):
    result = []
    for i in range(len(input_sent)):
        sent = input_sent[i].split()
        if sent[-2] == "where" or sent[-2] == "when" or sent[-2] == "what":
            sent.insert(0, "does")
            sent.insert(0, sent[-2])
            sent.pop(-2)
        sent_str = ' '.join(sent)
        result.append(sent_str)

    return result

print("Wh-moved questions: ", moveWh(questions), "\n")
