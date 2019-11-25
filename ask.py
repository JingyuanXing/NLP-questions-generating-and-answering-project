
"""
Group No Logic Please

Code for ask generation, will be compiled into executables

"""

import sys
import spacy
import re
# from QCheck import sentenceCheck
from spacy.symbols import nsubj, VERB, AUX, NOUN, attr, dobj, agent, amod, prep, advmod, auxpass, PROPN

import spacy


import nltk
nltk.download('wordnet', quiet=True)
nltk.download('averaged_perceptron_tagger', quiet=True)

import nlpaug.augmenter.word as naw
aug = naw.SynonymAug()

nlp = spacy.load("en_core_web_lg")


# grcheck
def questionToSentence(question):
    if question[0] in ["Who", "What", "Where", "When", "Why", "How"]:
        question = question[1:]


# this function is called by function sentenceCheck
# INPUT: spacy sentence, spacy verb token; OUTPUT: integer denoting how many points to subtract from score
def subjectVerbAgree(d, verb):
    if d["singleSubject"] == False and d["pluralSubject"] == False and d["expletive"] == None:
        return 2

    if d["pluralSubject"] == False and d["subjectWord"] not in ["you", "i"]:  # 3rd person singular
        if verb.tag_ in ["VBD", "VBZ"] and d["aux"] == None:  # i.e. "She wrote.", "She writes."
            return 0
        elif verb.tag_ == "VBN":  # "written"
            if d["aux"] in ["was", "is", "had", "has", "will have", "had been", "has been", "will have been"]:
                return 0
        elif verb.tag_ == "VBP":  # 3rd person plural - "They write."
            return 2
        elif verb.tag_ == "VB" and d["pluralVerb"] and d["aux"] == "to":  # i.e. "He is going to write."
            return 0
        elif verb.tag_ == "VB" and d["aux"] in ["will", "did"]:  # i.e. "He will write."
            return 0
        elif verb.tag_ == "VBG":  # "It is writing."
            if d["aux"] in ["was", "is", "did", "will be", "had been", "will have been"]:
                return 0
        return 2

    if d["pluralSubject"] == False and d["subjectWord"] == "you":
        if verb.tag_ == "VBZ":
            return 2
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
        return 2

    if d["pluralSubject"] == False and d["subjectWord"] == "i":
        if verb.tag_ == "VBZ":
            return 2
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
        return 2

    if d["pluralSubject"] == True:
        if verb.tag_ == "VBZ":  # "writes"
            return 2
        elif verb.tag_ == "VBD" and d["aux"] == None:  # "They wrote"
            return 0
        elif verb.tag_ == "VBP":  # "They write."
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
        return 2
    return 2


# Called by sentenceCheck
# INPUT: a spacy question; OUTPUT: a spacy sentence
def questionCheck(tokQ):
    questionWord = None
    for i in range(len(tokQ)):
      if tokQ[i].text.lower() in ["who", "what", "where", "when", "why", "how"]:
        tokQ = tokQ[(i+1):]
        questionWord = tokQ[i].text
    if questionWord == None:
      for i in range(len(tokQ)):
        if tokQ[i].text.lower() in ["is", "was", "does", "did"]:
          tokQ = tokQ[(i+1):]
          questionWord = tokQ[i].text
    temp = ""
    foundVerb = False
    for i in range(1, len(tokQ) - 1):

        if tokQ[i].pos_ == "VERB":
            foundVerb = True
            temp = temp + questionWord + " " + tokQ[i].text + " "
        else:
            temp = temp + tokQ[i].text + " "
    if not foundVerb:
        temp = temp + questionWord
    temp += "."
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

        if count < 4 and a.ents == []:
            score -= 5

        if count < 3 and a.ents == []:
            score -= 5

        if a[-1].tag_ != ".": score -= 1

        for tok in a:
            if tok.pos_ != "VERB" and tok.dep_ == "ROOT":
                score -= 2

        d = dict()
        d["subjectWord"] = None  # this is only used to record if the subject is "I" or "you" (since they used different verb conjugations)
        d["singleSubject"] = False
        # d["expletive"] = None #"this", "there", "that", etc.
        d["pluralSubject"] = False
        # d["tense"] = None
        d["aux"] = None
        d["verb"] = None
        d["pluralVerb"] = False
        d["compound"] = False  # does the sentence have multiple subjects/verbs?
        d["conj_AND"] = False
        d["object"] = False

        for i in range(len(a)):

            if a[i].text == "and" and (d["singleSubject"] or d["pluralSubject"]):
                d["conj_AND"] = True

            if a[i].pos_ in ["NOUN", "PROPN", "PRON"] and (
                    d["singleSubject"] != False or d["pluralSubject"] != False) and d["verb"] != None and not (
                    a[i].dep_ == "conj" and d["object"] == True) and a[i].dep_ not in ["dobj", "obj",
                                                                                       "pobj"]:  # already a simple phrase
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

            elif a[i].pos_ == "VERB":  # we found a verb - can be used for sentences with one or multiple verbs
                if d["verb"] != None:
                    d["pluralVerb"] = True
                if (d["singleSubject"] or d["pluralSubject"]):
                    score -= subjectVerbAgree(d, a[i])
                    if score <= 0:
                        score = 0
                        break
                    d["verb"] = a[i]

            elif a[i].pos_ in ["NOUN", "PROPN", "PRON"] and a[i].dep_ not in ["dobj", "obj", "pobj"] and not (
                    a[i].dep_ == "conj" and d["object"] == True):  # we found a subject word
                if (d["singleSubject"] or a[i].text.lower() in ["they", "we"]) and d["conj_AND"] == True:
                    d["singleSubject"] = False
                    d["pluralSubject"] = True  # i.e. "(x and y) went traveling last weekend..."
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

# Called by sentenceCheck
# INPUT: a spacy question; OUTPUT: a spacy sentence
def questionCheck(tokQ):
    if tokQ[0].text in ["Who", "What", "Where", "When", "Why", "How"]:
        tokQ = tokQ[1:]
    questionWord = tokQ[0].text
    temp = ""
    foundVerb = False
    for i in range(1, len(tokQ) - 1):

        if tokQ[i].pos_ == "VERB":
            foundVerb = True
            temp = temp + questionWord + " " + tokQ[i].text + " "
        else:
            temp = temp + tokQ[i].text + " "
    if not foundVerb:
        temp = temp + questionWord
    temp += "."
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
        d[
            "subjectWord"] = None  # this is only used to record if the subject is "I" or "you" (since they used different verb conjugations)
        d["singleSubject"] = False
        # d["expletive"] = None #"this", "there", "that", etc.
        d["pluralSubject"] = False
        # d["tense"] = None
        d["aux"] = None
        d["verb"] = None
        d["pluralVerb"] = False
        d["compound"] = False  # does the sentence have multiple subjects/verbs?
        d["conj_AND"] = False
        d["object"] = False

        for i in range(len(a)):

            if a[i].text == "and" and (d["singleSubject"] or d["pluralSubject"]):
                d["conj_AND"] = True

            if a[i].pos_ in ["NOUN", "PROPN", "PRON"] and (
                    d["singleSubject"] != False or d["pluralSubject"] != False) and d["verb"] != None and not (
                    a[i].dep_ == "conj" and d["object"] == True) and a[i].dep_ not in ["dobj", "obj",
                                                                                       "pobj"]:  # already a simple phrase
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

            elif a[i].pos_ == "VERB":  # we found a verb - can be used for sentences with one or multiple verbs
                if d["verb"] != None:
                    d["pluralVerb"] = True
                if (d["singleSubject"] or d["pluralSubject"]):
                    score -= subjectVerbAgree(d, a[i])
                    if score <= 0:
                        score = 0
                        break
                    d["verb"] = a[i]

            elif a[i].pos_ in ["NOUN", "PROPN", "PRON"] and a[i].dep_ not in ["dobj", "obj", "pobj"] and not (
                    a[i].dep_ == "conj" and d["object"] == True):  # we found a subject word
                if (d["singleSubject"] or a[i].text.lower() in ["they", "we"]) and d["conj_AND"] == True:
                    d["singleSubject"] = False
                    d["pluralSubject"] = True  # i.e. "(x and y) went traveling last weekend..."
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


def question_generate(relations):
    questions = []
    for r in relations:
        subj = r[0]
        fill = r[1]
        obj = r[2]
        subject_nlp = nlp(subj)
        classified_subj = [(X.text, X.label_) for X in subject_nlp.ents]
        #print(classified_subj)
        fill_nlp = nlp(fill)
        object_nlp = nlp(obj)
        classified_obj = [(X.text, X.label_) for X in object_nlp.ents]
        #print(classified_obj)

        past_flag = False
        third_flag = False
        for f in fill_nlp:
            if (nlp.vocab.morphology.tag_map[f.tag_].get('Tense_past') != None):
                past_flag = True
            if (f.tag_ == "VBZ"):
                third_flag = True

        # compile the words in the subjects and objects into phrases.
        phrase = " ".join(r)
        # print(phrase + "\n")

        # general question with 'what' replacement.
        phrase_no_subj = " ".join([r[1], r[2]])
        phrase_no_obj = " ".join([r[0], r[1]])
        questions.append("What " + phrase_no_subj + "?")



        # Question based on auxiliary verb
        # including binary and what - question
        for vocab in nlp(phrase):
            # print(vocab.dep_)
            if vocab.dep_ == "aux" and (vocab.text != "to"):
                b = vocab.text + " " + re.sub(r'\b%s\b' % re.escape(vocab.text), "", phrase, 1) + "?"
                o = "What " + vocab.text + " " + re.sub(r'\b%s\b' % re.escape(vocab.text), "", phrase_no_obj, 1) + "?"
                b = b[0].upper() + b[1:]
                questions.append(b)
                questions.append(o)

        # get the pronouns
        for s in subject_nlp:
            if (s.pos_ == "PRON"):
                questions.append(re.sub(r'\b%s\b' % re.escape(s.text), "who", phrase, 1) + "?")
        for o in object_nlp:
            if (o.pos_ == "PRON"):
                questions.append(re.sub(r'\b%s\b' % re.escape(o.text), "who", phrase, 1) + "?")


        # specific questions targeting the named entity
        for word_tuple in classified_subj + classified_obj:
            if (word_tuple[1] == 'PERSON'):
                questions.append(re.sub(r'\b%s\b' % re.escape(word_tuple[0]), "who", phrase, 1) + "?")
            if (word_tuple[1] == 'LOC'):
                questions.append(re.sub(r'\b%s\b' % re.escape(word_tuple[0]), "where", phrase, 1) + "?")
            if (word_tuple[1] == 'GPE'):
                questions.append(re.sub(r'\b%s\b' % re.escape(word_tuple[0]), "what country or city", phrase, 1) + "?")
            if (word_tuple[1] == 'NORP'):
                questions.append(re.sub(r'\b%s\b' % re.escape(word_tuple[0]), "what country", phrase, 1) + "?")
            if (word_tuple[1] == 'EVENT'):
                questions.append(re.sub(r'\b%s\b' % re.escape(word_tuple[0]), "where", phrase, 1) + "?")
            if (word_tuple[1] == 'LANGUAGE'):
                questions.append(re.sub(r'\b%s\b' % re.escape(word_tuple[0]), "what lanuage", phrase, 1) + "?")
            if (word_tuple[1] == 'DATE' or word_tuple[1] == 'TIME'):
                questions.append(re.sub(r'\b%s\b' % re.escape(word_tuple[0]), "at what time", phrase, 1) + "?")
            if (word_tuple[1] == 'PERCENT'):
                questions.append(re.sub(r'\b%s\b' % re.escape(word_tuple[0]), "of what persent", phrase, 1) + "?")
            if (word_tuple[1] == 'ORG'):
                questions.append(re.sub(r'\b%s\b' % re.escape(word_tuple[0]), "what organization", phrase, 1) + "?")

    # print("************** All questions **************")
    # print(questions)

    final = []
    for q in questions:
        # augment it for 2 times
        a1 = aug.augment(q)
        a2 = aug.augment(q)
        # capitalize the first character
        final.append(a1[0].upper() + a1[1:])
        final.append(a2[0].upper() + a2[1:])
    return list(set(final))
    # return questions


########################################
#######################################
from spacy.symbols import nsubj, VERB, AUX, NOUN, attr, dobj, agent, amod, prep, advmod, auxpass, PROPN


def find_verb_phrase(vb_idx, vb_result_idx, result_idx):
    res = [vb_result_idx]
    left_result_idx = vb_result_idx - 1
    last_idx = vb_idx
    # check left
    while left_result_idx >= 0:
        left_idx = result_idx[left_result_idx]
        if last_idx - left_idx > 1:
            break
        else:
            last_idx = left_idx
            res.append(left_result_idx)
            left_result_idx -= 1
    # check right
    right_result_idx = vb_result_idx + 1
    last_idx = vb_idx
    while right_result_idx < len(result_idx):
        right_idx = result_idx[right_result_idx]
        if right_idx - last_idx > 1:
            break
        else:
            last_idx = right_idx
            res.append(right_result_idx)
            right_result_idx += 1
    return res


# given a verb, generate the verb phrase around the verb
def generate_verb_phrase(vb):
    result = []
    result_idx = []
    final_result = []
    verb_idx = None
    keep_list = []
    for ind, word in enumerate(vb.subtree):
        if ((word.dep in [agent, prep, advmod, auxpass]) and word.head == vb) or word == vb:
            result.append(word)
            result_idx.append(ind)
            if word == vb:
                verb_idx = ind
                verb_result_idx = len(result) - 1
    # to remove incontinuous component of verb phrase
    keep_list = find_verb_phrase(verb_idx, verb_result_idx, result_idx)
    for i in sorted(keep_list):
        final_result.append(result[i])
    return final_result


# given the verb phrase and the verb and the main relations, try to find the corresponding subject part of the relation
def generate_subject_phrase(vb, vp, rel):
    sub_list = []
    if vb.head.pos == VERB:
        for word in vb.subtree:
            if word == vp[0] and (word == vb or word.head == vb): break
            if word.head == vb and word not in vp and word.pos not in [NOUN, PROPN]: continue
            sub_list.append(word)
    else:
        for word in vb.head.subtree:
            if word == vp[0] and (word == vb or word.head == vb): break
            if word.head == vb and word not in vp and word.pos not in [NOUN, PROPN]: continue
            sub_list.append(word)

    return sub_list


def generate_object_phrase(vb, vp):
    ob_list = []
    okay_to_insert_object = False
    for ind, word in enumerate(vb.subtree):
        if word in vp and (word.head == vb or word == vb):
            okay_to_insert_object = True
        elif word not in vp and okay_to_insert_object:
            ob_list.append(word)
    return ob_list

###################################################
###################################################
# class NLP Paragraph

from spacy.symbols import nsubj, VERB, AUX, NOUN, attr, dobj, agent, amod, prep, advmod, auxpass, nsubjpass, acl


class NLPparagraph():

    def __init__(self, paragraph, p_number=1):
        self.doc = nlp(paragraph)
        self.sentences = [sent.string.strip() for sent in self.doc.sents]
        self.relationships = []

    def __repr__(self):
        return self.doc.text

    def __str__(self):
        return self.doc.text

    def sent_relation_extraction(self, sent_ind=0):
        cur_sent = nlp(self.sentences[sent_ind])
        # high level extraction (simply np vp extraction for sentence)

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
            relationship = (subject_list, [root], object_list, 0)
            self.relationships.append(relationship)

        # extraction based on each verb

        verbs = [token for token in cur_sent if token.pos == VERB or token.pos == AUX]
        for verb in verbs:
            subject_list = []
            object_list = []
            vp_list = generate_verb_phrase(verb)
            if verb.head == verb or verb.dep == amod or len(vp_list) == 0 or (verb.head.pos != NOUN and verb.head.pos != PROPN):
                continue
            else:
                subject_list = generate_subject_phrase(verb, vp_list, self.relationships)
                object_list = generate_object_phrase(verb, vp_list)
                if_passive = 0 if verb.dep != acl else 1
                if len(object_list) > 0 and len(subject_list) > 0:
                    relationship = (subject_list, vp_list, object_list, if_passive)
                    self.relationships.append(relationship)
        return self.relationships

    def paragraph_relation_extraction(self):
        for sent_idx in range(len(self.sentences)):
            self.sent_relation_extraction(sent_idx)
        return self.relationships


class article():

    def __init__(self, text):
        self.text = text
        raw_paragraphs = text.split('\n')
        self.paragraphs = []
        self.title = raw_paragraphs[0]
        self.relations = []
        for p in raw_paragraphs:
            if len(p.split(' ')) > 10:
                NLPP = NLPparagraph(p)
                self.paragraphs.append(NLPP)

    def __repr__(self):
        return self.text

    def __str__(self):
        return self.text

    def relation_extraction(self):
        self.relations = []
        for pa in self.paragraphs:
            pa_relations = pa.paragraph_relation_extraction()
            for j in pa_relations:
                self.relations.append(j)
        for i, rel in enumerate(self.relations):
            obj = rel[2]
            verb = rel[1]
            sub = rel[0]
            passive = rel[3]
            newobj = ' '.join([elem.text for elem in obj])
            newsub = ' '.join([elem.text for elem in sub])
            if passive:
                newverb_list = [elem.text for elem in verb]
                newverb_list.insert(0, "is")
                newverb = ' '.join(newverb_list)
            else:
                newverb = ' '.join([elem.text for elem in verb])
            self.relations[i] = [newsub, newverb, newobj]
        return self.relations



def main():
    if len(sys.argv) != 3:
        print("please input article path and number of questions")
        sys.exit(1)
    article_path, nQ = sys.argv[1], int(sys.argv[2])
    r = open(file=article_path, mode = "r", encoding="utf-8")
    text1 = r.read()
    art1 = article(text1)
    art1.relation_extraction()
    qs = question_generate(art1.relations)
    q_nlp = [nlp(q) for q in qs]
    q_ranks = sentenceCheck(q_nlp, True)
    questions = [(rank, q) for rank, q in sorted(zip(q_ranks, qs), reverse=True)]
    if nQ >= len(questions):
        for i, line in enumerate(questions):
            print('A{}. {} {}'.format(i+1, line[1].strip(), line[0]))
    else:
        for i, line in enumerate(questions[0:nQ]):
            print('A{}. {} {}'.format(i+1, line[1].strip(), line[0]))


if __name__ == '__main__':
    main()

