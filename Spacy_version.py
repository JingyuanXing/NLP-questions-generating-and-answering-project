import spacy

nlp = spacy.load("en_core_web_sm")
r = open(file = "Development_data/set1/a1.txt", mode = "r")
text = r.read()

################################################################################

paragraphs = text.split('\n')
current_para = paragraphs[3]
doc = nlp(current_para)
sentences = [sent.string.strip() for sent in doc.sents]
#print(sentences)

################################################################################

from spacy.symbols import nsubj, VERB, AUX, NOUN, attr, dobj, agent, amod, prep, advmod, auxpass

#given a verb, generate the verb phrase around the verb
def generate_verb_phrase(vb):
    result = []
    result_idx = []
    verb_idx = None
    delete_list = []
    for ind, word in enumerate(vb.subtree):
        if ((word.dep in [agent, prep, advmod, auxpass]) and word.head == vb) or word == vb:
            result.append(word)
            result_idx.append(ind)
            if word == vb:
                verb_idx = ind
                verb_result_idx = len(result)-1
    # to remove incontinuous component of verb phrase
    for i in range(len(result)):
        if verb_idx - result_idx[i] != verb_result_idx - i:
            delete_list.append(i)
    if len(delete_list) > 0 and result[0] != vb:
        maxi = max(delete_list)
        result = result[maxi+1:]
    return result

#given the verb phrase and the verb and the main relations, try to find the corresponding subject part of the relation
def generate_subject_phrase(vb, vp, rel):
    sub_list = []
    if vb.head.pos == VERB:
        for word in vb.subtree:
            if word == vp[0] and (word==vb or word.head==vb): break
            if word.head == vb and word not in vp and word.pos != NOUN: continue
            sub_list.append(word)
    else:
        for word in vb.head.subtree:
            if word == vp[0] and (word==vb or word.head==vb): break
            if word.head == vb and word not in vp and word.pos != NOUN: continue
            sub_list.append(word)
    return sub_list

#given the verb phrase and the verb and the main relations, try to find the corresponding object part of the relation
def generate_object_phrase(vb, vp):
    ob_list = []
    okay_to_insert_object = False
    for ind, word in enumerate(vb.subtree):
        if word in vp and (word.head == vb or word == vb): 
            okay_to_insert_object = True
        elif word not in vp and okay_to_insert_object:
            ob_list.append(word)
    return ob_list
    


# relationships = []
# verbs = [token for token in cur_sent if token.pos == VERB or token.pos == AUX]

# for verb in verbs:
#     print(generate_verb_phrase(verb))

################################################################################

from spacy.symbols import nsubj, VERB, AUX, NOUN, attr, dobj, agent, amod, prep, advmod, auxpass, nsubjpass


class NLPparagraph():
    
    def __init__(self, paragraph, p_number=1):
        self.doc = nlp(paragraph)
        self.sentences = [sent.string.strip() for sent in self.doc.sents] 
        self.relationships = []
        

    def sent_relation_extraction(self, sent_ind=0):
        cur_sent = nlp(self.sentences[sent_ind])
        #high level extraction (simply np vp extraction for sentence)
        
        root = [token for token in cur_sent if token.head == token][0]
        possible_phrases = list(root.children)
        subject_list = []
        object_list = []
        for possible_phrase in possible_phrases:
            if possible_phrase.dep in [nsubj, nsubjpass] and (root.pos == VERB or root.pos == AUX):
                for i in possible_phrase.subtree:
                    subject_list.append(i)
            elif (possible_phrase.dep == dobj or possible_phrase.dep == attr) and (root.pos == VERB or root.pos == AUX):
                for j in possible_phrase.subtree:
                    object_list.append(j)
            
        if len(object_list) != 0 and len(subject_list) != 0:
            relationship = (subject_list, [root],  object_list)
            self.relationships.append(relationship)
        
        
        #extraction based on each verb
        
        verbs = [token for token in cur_sent if token.pos == VERB or token.pos == AUX]
        for verb in verbs:
            subject_list = []
            object_list = []
            vp_list = generate_verb_phrase(verb)
            if verb.head == verb or verb.dep == amod:
                continue
            else:
                subject_list = generate_subject_phrase(verb, vp_list, relationships)
                object_list = generate_object_phrase(verb, vp_list)
                if len(object_list) > 0 and len(subject_list) > 0:
                    relationship = (subject_list, vp_list, object_list)
                    self.relationships.append(relationship)
        return self.relationships



a = NLPparagraph(current_para)
print(a.sent_relation_extraction())


################################################################################ you can ignore everything below

cur_sent = nlp(sentences[3])
for token in cur_sent:
    print(token.text,token.pos_, token.tag_, token.dep_, 
          token.head.text, token.head.pos_)

spacy.displacy.serve(cur_sent, style = "dep")

################################################################################

from spacy.symbols import nsubj, VERB, AUX, NOUN, attr, dobj, agent, amod, prep, advmod, auxpass

relationships = []
verbs = [token for token in cur_sent if token.pos == VERB or token.pos == AUX]

for verb in verbs:
    subject_list = []
    object_list = []
    vp_list = generate_verb_phrase(verb)
    if verb.head == verb or verb.dep == amod:
        continue
    else:
        subject_list = generate_subject_phrase(verb, vp_list, relationships)
        object_list = generate_object_phrase(verb, vp_list)
        if len(object_list) > 0 and len(subject_list) > 0:
            relationship = (subject_list, vp_list, object_list)
            relationships.append(relationship)

print(relationships)

################################################################################

####andy's trial code 
from spacy.symbols import nsubj, VERB, AUX, NOUN, attr, dobj, auxpass, nsubjpass

relationships = []
root = [token for token in cur_sent if token.head == token][0]
print(root.pos == AUX)
possible_phrases = list(root.children)
subject_list = []
object_list = []
for possible_phrase in possible_phrases:
    if possible_phrase.dep in [nsubj, nsubjpass] and (root.pos == VERB or root.pos == AUX):
        for i in possible_phrase.subtree:
            subject_list.append(i)
    elif (possible_phrase.dep == dobj or possible_phrase.dep == attr) and (root.pos == VERB or root.pos == AUX):
        for j in possible_phrase.subtree:
            object_list.append(j)
            
            
if len(object_list) != 0 and len(subject_list) != 0:
    relationship = (subject_list, [root],  object_list)
    relationships.append(relationship)
    
print(relationships)

################################################################################

## Isabel's code (1)

import spacy
#!pip install textacy
import textacy

def questionToSentence(question):
  if question[0] in ["Who", "What", "Where", "When", "Why", "How"]:
    question = question[1:]
    

#this function is called by function sentenceCheck
#INPUT: spacy sentence, spacy verb token; OUTPUT: integer denoting how many points to subtract from score
def subjectVerbAgree(d, verb):

  if d["singleSubject"] == False and d["pluralSubject"] == False and d["expletive"] == None:
    return 4

  if d["pluralSubject"] == False and d["subjectWord"] not in ["you", "i"]: #3rd person singular
    if verb.tag_ in ["VBD", "VBZ"] and d["aux"] == None: #i.e. "She wrote.", "She writes."
      return 0
    elif verb.tag_ == "VBN": #"written"
      if d["aux"] in ["was", "is", "had", "has", "will have", "had been", "has been", "will have been"]:
        return 0
    elif verb.tag_ == "VBP": #3rd person plural - "They write."
      return 4
    elif verb.tag_ == "VB" and d["pluralVerb"] and d["aux"] == "to": # i.e. "He is going to write."
      return 0
    elif verb.tag_ == "VB" and d["aux"] in ["will", "did"]: # i.e. "He will write."
      return 0
    elif verb.tag_ == "VBG": #"It is writing."
      if d["aux"] in ["was", "is", "did", "will be", "had been", "will have been"]:
        return 0
    return 4

  if d["pluralSubject"] == False and d["subjectWord"] == "you":
    if verb.tag_ == "VBZ":
      return 4
    elif verb.tag_ == "VBD" and d["aux"] == None:
      return 0
    elif verb.tag_ == "VBN":
      if d["aux"] in ["were", "are", "had", "have", "will have"]:
        return 0
    elif verb.tag_ == "VBP":
      if d["aux"] in [None, "will", "did"]:
        return 0
    elif verb.tag_ == "VB" and d["pluralVerb"] and d["aux"] == "to":
      return 0
    elif verb.tag_ == "VB" and d["aux"] in ["will", "did"]:
      return 0
    elif verb.tag_ == "VBG":
      if d["aux"] in [None, "did", "were", "are", "will be", "had been", "have been", "will have been"]:
        return 0
    return 4

  if d["pluralSubject"] == False and d["subjectWord"] == "i":
    if verb.tag_ == "VBZ":
      return 4
    elif verb.tag_ == "VBD" and d["aux"] == None:
      return 0
    elif verb.tag_ == "VBN":
      if d["aux"] in ["was", "am", "had", "have", "will have"]:
        return 0
    elif verb.tag_ == "VBP":
      if d["aux"] in [None, "will", "did"]:
        return 0
    elif verb.tag_ == "VB":
      if (d["pluralVerb"] and d["aux"] == "to"):
        return 0
    elif verb.tag_ == "VBG":
      if d["aux"] in ["did", "was", "am", "will be", "had been", "have been", "will have been"]:
        return 0
    return 4

  if d["pluralSubject"] == True:
    if verb.tag_ == "VBZ": #"writes"
      return 4
    elif verb.tag_ == "VBD" and d["aux"] == None: #"They wrote"
      return 0
    elif verb.tag_ == "VBP": #"They write."
      if d["aux"] in [None, "will"]:
        return 0
    elif verb.tag_ == "VB":
      if d["pluralVerb"] and d["aux"] in ["to", "will"]:
        return 0
    elif verb.tag_ == "VBG":
      if d["aux"] in ["were", "are", "was", "will be", "had been", "have been", "will have been"]:
        return 0
    elif verb.tag_ == "VBN":
      if d["aux"] in ["were", "had", "have", "will have"]:
        return 0
    return 4
  return 4

#Called by sentenceCheck
#INPUT: a spacy question; OUTPUT: a spacy sentence
def questionCheck(tokQ):
  if tokQ[0].text in ["Who", "What", "Where", "When", "Why", "How"]:
    tokQ = tokQ[1:]
  questionWord = tokQ[0].text
  temp = ""
  foundVerb = False
  for i in range(1, len(tokQ)-1):

    if tokQ[i].pos_ == "VERB":
      foundVerb = True
      temp = temp + questionWord + " " + tokQ[i].text + " "
    else:
      temp = temp + tokQ[i].text + " "
  if not foundVerb:
    temp = temp + questionWord 
  temp += "."
  nlp = spacy.load("en_core_web_sm")
  newTokSent = nlp(temp)
  return newTokSent
        
def sentenceCheck(lst, isQues):
  scores = []
  for a in lst:
    if isQues:
      a = questionCheck(a)
    count = 0
    for tok in a:
      count += 1

    score = 10
    if a.text == "":
      return 0

    if count < 4:
      score -= 5

    if count < 3:
      score -= 5


    if a[-1].tag_ != ".": score -= 1


    for tok in a:
      if tok.pos_ != "VERB" and tok.dep_ == "ROOT":
        score -= 2

    d = dict()
    d["subjectWord"] = None #this is only used to record if the subject is "I" or "you" (since they used different verb conjugations)
    d["singleSubject"] = False
    #d["expletive"] = None #"this", "there", "that", etc.
    d["pluralSubject"] = False
    #d["tense"] = None
    d["aux"] = None
    d["verb"] = None
    d["pluralVerb"] = False
    d["compound"] = False #does the sentence have multiple subjects/verbs?
    d["conj_AND"] = False

    for i in range(len(a)):

      if a[i].text == "and" and (d["singleSubject"] or d["pluralSubject"]):
        d["conj_AND"] = True

      if a[i].pos_ in ["NOUN", "PROPN", "PRON"] and (d["singleSubject"] != False or d["pluralSubject"] != False) and d["verb"] != None and not (a[i].dep_ == "conj" and d["object"] == True) and a[i].dep_ not in ["dobj", "obj", "pobj"]: #already a simple phrase
        d["compound"] = True
        d["verb"] = d["expletive"] = d["aux"] = d["subjectWord"] = None
        d["singleSubject"] = d["pluralSubject"] = d["conj_AND"] = d["object"] = False
        if a[i].text.lower() in ["they", "we"]:
          d["pluralSubject"] = True
        else:
          d["singleSubject"] = True
          if a[i].text.lower() in ["you", "i"]:
            d["subjectWord"] = a[i].text.lower()

      if a[i].dep_ in ["dobj", "obj", "pobj"]:
        d["object"] = True
    
      if a[i].dep_ in ["aux", "auxpass"]:
        if d["aux"] != None:
          d["aux"] = d["aux"] + " " + a[i].text.lower()
        else:
          d["aux"] = a[i].text.lower()

      elif a[i].pos_ == "VERB": #we found a verb - can be used for sentences with one or multiple verbs
        if d["verb"] != None:
          d["pluralVerb"] = True
        if (d["singleSubject"] or d["pluralSubject"]):
          score -= subjectVerbAgree(d, a[i])
          if score <= 0:
            score = 0
            break
          d["verb"] = a[i]

      elif a[i].pos_ in ["NOUN", "PROPN", "PRON"] and a[i].dep_ not in ["dobj", "obj", "pobj"] and not (a[i].dep_ == "conj" and d["object"] == True): #we found a subject word
        if (d["singleSubject"] or a[i].text.lower() in ["they", "we"]) and d["conj_AND"] == True:
          d["singleSubject"] = False
          d["pluralSubject"] = True # i.e. "(x and y) went traveling last weekend..."
        else:
          d["singleSubject"] = True
          if a[i].text.lower() in ["you", "i"]:
            d["subjectWord"] = a[i].text.lower()

      if a[-2].tag_ == "CC":
        score -= 3

    if d["verb"] == None:
      score -= 2
    if d["singleSubject"] == False and d["pluralSubject"] == False:
      score -= 2
    if score < 0: score = 0
    scores.append(score)
  return scores

################################################################################

## Isabel's code (2)

import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

#INPUT: two vectors; OUTPUT: a float
def cosineSimilarity(a,b):

  dotProd = 0
  for i in range(min(len(a), len(b))):
    dotProd += (a[i]*b[i])
  if len(a) > len(b):
    for j in range(len(b), len(a)):
      dotProd += a[j]
  if len(b) > len(a):
    for j in range(len(a), len(b)):
      dotProd += b[j]
  
  return dotProd / (len(a)*len(b))

#INPUT: list of questions, list of sentences; OUTPUT: a dictionary matching each question with the best fitting sentence from the article
def tfidf(questions, sentences):

  vectorizer = TfidfVectorizer()

  x = vectorizer.fit(questions)
  questionArray = x.transform(questions).toarray()
  y = vectorizer.fit(sentences)
  sentenceArray = y.transform(sentences).toarray()

  questionSentenceMatch = dict()

  for ques in questionArray:
    maxCS = 0
    bestSent = sentences[0]
    for i in range(len(sentenceArray)):
      cs = cosineSimilarity(q, sentenceArray[i])
      if cs > maxCS:
        maxCS = cs
        bestSent = sentences[i]
    questionSentenceMatch[ques] = (bestSent, maxCS)
  return questionSentenceMatch

################################################################################

# Isabel's code (3)
# INPUT: STRING "where" question, STRING full sentence from the document containing the answer
# OUTPUT: STRING answer

def whereQues(whereQues, ansSentence):
  nlp = spacy.load("en_core_web_sm")
  question = nlp(whereQues)
  sentence = nlp(ansSentence)
  NERlst = []
  questionVerb = None
  questionSubject = None
  for word in question:
    print(word, word.pos_, word.dep_)
    if word.pos_ == "VERB":
      questionVerb = word
    if word.dep_ in ["nsubj", "nsubjpass"] and questionSubject == None:
      questionSubject = word.text.lower()
  print(questionSubject, questionVerb)
  for tok in sentence.ents:
    if tok.label_ == "GPE":
      NERlst.append(tok.text)
  prepPhrase = []
  loc = []
  prep = None
  subject = []
  verb = []
  for word in sentence:
    print(word.text, word.pos_, word.tag_, word.dep_, word.head.text)
    if prep != None and prepPhrase != None and (word.pos_ == "CCONJ" or word.tag_ == "CC"): #going into second clause of the sentence, but we have our answer already
      break
    elif word.dep_ in ["nsubj", "nsubjpass"] and word.text == questionSubject:
      subject.append(word.text)
    elif word.pos_ == "VERB" and word.dep_ == "auxpass" and word.head.lemma_ == questionVerb.lemma_:
      verb.append(word.text)
    elif word.pos_ == "VERB" and word.lemma_ == questionVerb.lemma_:
      verb.append(word.text)
      print("VERB: ____", verb)
    elif word.pos_ in ["DET", "ADJ"] and word.head.text == questionSubject:
      subject.append(word.text)
    elif word.pos_ == "ADP" and word.head.text in verb:
      prep = word.text
      prepPhrase.append(prep)
    elif word.head.text == prep:
      prepPhrase.append(word.text)

  if subject == None or verb == None or prep == None:
    print(subject, verb, prep)
    return sentence
  
  question = question[1:] #get rid of "where"
  if question[0].pos_ == "VERB":
    question = question[1:]
  ans = " ".join(subject) + " " + " ".join(verb) + " " + " ".join(prepPhrase) + "."
  return ans

################################################################################

import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import StanfordNERTagger

import spacy
from spacy import displacy
from spacy.matcher import Matcher

from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()

import re

spacy_nlp = spacy.load("en_core_web_sm")
matcher = Matcher(spacy_nlp.vocab)

jar = 'stanford-ner/stanford-ner.jar'
# model = 'stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz'
model = 'stanford-ner/classifiers/english.muc.7class.distsim.crf.ser.gz'
# model = 'stanford-ner/classifiers/wikigold.conll.ser.gz'
# model = 'stanford-ner/classifiers/sentiment.ser.gz'
st = StanfordNERTagger(model, jar, encoding='utf-8')


question = "When was the first known labor strike occurred ?"


r = open(file = 'Development_data/set1/a3.txt', mode = "r")
text = r.read()

text = text.replace("(", ",")
text = text.replace(")", ",")
text_lines = text.splitlines()
# print(text_lines)

sent_tokens = []
for line in text_lines:
    sent_tokens.extend(sent_tokenize(line))
# print(len(sent_tokens))
sents_tagged = [nltk.pos_tag(word_tokenize(sent)) for sent in sent_tokens]
# print(len(sents_tagged))


question_tokens = word_tokenize(question)
question_pattern = [];
for t in question_tokens:
    question_pattern.extend([{'LEMMA': t, 'OP': '?'}])

print(question_pattern)
matcher.add("question_pattern", None, question_pattern)

matched_sentence = ""
max_match_length = 0;

for sent_token in sent_tokens:
    this_length = 0
    doc = spacy_nlp(sent_token)
    matches = matcher(doc)
    for match_id, start, end in matches:
        # string_id = nlp.vocab.strings[match_id]  # Get string representation
        span = doc[start:end]  # The matched span
        this_length += len(span.text)        
        # if (span.text != ""):
        #     print(start, end, span.text)
    if (this_length > max_match_length):
        max_match_length = this_length
        matched_sentence = sent_token

# print()
print(matched_sentence)

################################################################################

import nlpaug.augmenter.word as naw
# aug = naw.WordNetAug()
# aug = naw.RandomWordAug()
aug = naw.ContextualWordEmbsAug()
# aug = naw.SynonymAug()

# Substitute type questions
# relations: [(subject: string, fill: string, object: string), (subject, fill, object)...]
# return: [question1: string, question2, question3...]
def question_generate(relations):
    questions = []
    for r in relations:
        # print(r)
        subj = r[0]
        obj = r[2]
        subject_nlp = nlp(subj)
        classified_subj = [(X.text, X.label_) for X in subject_nlp.ents]
        object_nlp = nlp(obj)
        classified_obj = [(X.text, X.label_) for X in object_nlp.ents]

        # compile the words in the subjects and objects into phrases.
        phrase = " ".join(r)

        # general question with 'what' replacement. 
        phrase_no_subj = " ".join([r[1], r[2]])
        phrase_no_obj = " ".join([r[0], r[1]])
        questions.append("What " + phrase_no_subj + "?")
        # questions.append(phrase_no_obj + " what " + "?")

        # for the what - substitiute of object, put What at front with
        questions.append("What " + r[1] + " " + r[0] + "?")

        # specific questions targeting the named entity
        for word_tuple in classified_subj + classified_obj:
            if (word_tuple[1] == 'PERSON'):
                questions.append(re.sub(r'\b%s\b' % word_tuple[0], "who", phrase, 1) + "?")
            if (word_tuple[1] == 'LOC'):
                questions.append(re.sub(r'\b%s\b' % word_tuple[0], "where", phrase, 1) + "?")
            if (word_tuple[1] == 'GPE'):
                questions.append(re.sub(r'\b%s\b' % word_tuple[0], "what country or city", phrase, 1) + "?")
            if (word_tuple[1] == 'NORP'):
                questions.append(re.sub(r'\b%s\b' % word_tuple[0], "what country", phrase, 1) + "?")    
            if (word_tuple[1] == 'EVENT'):
                questions.append(re.sub(r'\b%s\b' % word_tuple[0], "where", phrase, 1) + "?")
            if (word_tuple[1] == 'LANGUAGE'):
                questions.append(re.sub(r'\b%s\b' % word_tuple[0], "what lanuage", phrase, 1) + "?")
            if (word_tuple[1] == 'DATE' or word_tuple[1] == 'TIME'):
                questions.append(re.sub(r'\b%s\b' % word_tuple[0], "at what time", phrase, 1) + "?")
            if (word_tuple[1] == 'PERCENT'):
                questions.append(re.sub(r'\b%s\b' % word_tuple[0], "of what percent", phrase, 1) + "?")
            if (word_tuple[1] == 'ORG'):
                questions.append(re.sub(r'\b%s\b' % word_tuple[0], "what organization", phrase, 1) + "?")   

    # print(questions)
    final = []
    for q in questions:
        a = aug.augment(q)
        final.append(a[0].upper() + a[1:])
    return final

    ############################################################################

    # question is a string
# text is a huge string
# return a dictionary {sentence -> score}. You might want to sort the ditionary to 
# get the highest one. 
# score should between 0 and 1, the higher the more similar.
def match_sentence(question, text):

    text_lines = text.splitlines()
    sent_tokens = []
    for line in text_lines:
        sent_tokens.extend(sent_tokenize(line))
    # print(len(sent_tokens))
    sents_tagged = [nltk.pos_tag(word_tokenize(sent)) for sent in sent_tokens]

    question_tokens = word_tokenize(question)
    question_pattern = [];
    for t in question_tokens:
        question_pattern.extend([{'LEMMA': t, 'OP': '?'}])

    print(question_pattern)
    matcher.add("question_pattern", None, question_pattern)

    matched_scores = {}
    max_length = 0

    for sent_token in sent_tokens:
        this_length = 0
        this_score = 0
        doc = spacy_nlp(sent_token)
        matches = matcher(doc)
        for match_id, start, end in matches:
            # string_id = nlp.vocab.strings[match_id]  # Get string representation
            span = doc[start:end]  # The matched span
            this_length += len(word_tokenize(span.text)) 
            # print(span.text)       
            # if (span.text != ""):
            #     print(start, end, span.text)
        max_length = max(this_length, max_length)
        matched_scores[sent_token] = this_length

    matched_scores = {k: (v/max_length) for k, v in matched_scores.items()}

    # from the default similarity function in spacy, find the simiarity score 
    # for the sentence, and then weight it with the matching score. 
    final_scores = {}

    question_nlp = spacy_nlp(question)
    for sent in matched_scores:
        similarity_score = spacy_nlp(sent).similarity(question_nlp)
        print(sent)
        print(matched_scores[sent])
        final_scores[sent] = (matched_scores[sent] + similarity_score) / 2

    return (final_scores)

################################################################################




