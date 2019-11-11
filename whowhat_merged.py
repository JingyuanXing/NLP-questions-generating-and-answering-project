import nltk
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize
from nltk.tag import StanfordNERTagger

import spacy
from spacy import displacy
from spacy.matcher import Matcher

from collections import Counter
import en_core_web_sm
import en_core_web_md

import re

import spacy
from spacy.symbols import nsubj, VERB, AUX, NOUN, attr, dobj, agent, amod, prep, advmod, auxpass, PROPN, DET, PRON


class Answering(object):
    def __init__(self, question, text):
        self.question = question
        self.text = text
        self.nlp = spacy.load("en_core_web_sm")

    ######################## returning selected sentence #######################

    def match_sentence(self):
        nlp = en_core_web_sm.load()
        spacy_nlp = spacy.load("en_core_web_md")
        matcher = Matcher(spacy_nlp.vocab)

        text_lines = self.text.splitlines()

        sent_tokens = []
        for line in text_lines:
            sent_tokens.extend(sent_tokenize(line))
        sents_tagged = [nltk.pos_tag(word_tokenize(sent)) for sent in sent_tokens]

        question_tokens = word_tokenize(self.question)
        question_pattern = [];
        for t in question_tokens:
            question_pattern.extend([{'LEMMA': t, 'OP': '?'}])

        matcher.add("question_pattern", None, question_pattern)

        matched_scores = {}
        max_length = 0

        for sent_token in sent_tokens:
            this_length = 0
            this_score = 0
            doc = spacy_nlp(sent_token)
            matches = matcher(doc)
            for match_id, start, end in matches:
                # string_id = nlp.vocab.strings[match_id]
                span = doc[start:end]  # The matched span
                this_length += len(word_tokenize(span.text)) 
            
            max_length = max(this_length, max_length)
            matched_scores[sent_token] = this_length

        matched_scores = {k: (v/max_length) for k, v in matched_scores.items()}

        # from the default similarity function in spacy, find the simiarity score 
        # for the sentence, and then weight it with the matching score. 
        final_scores = {}

        question_nlp = spacy_nlp(self.question)
        for sent in matched_scores:
            similarity_score = spacy_nlp(sent).similarity(question_nlp)
            final_scores[sent] = (matched_scores[sent] + similarity_score) / 2

        return (final_scores)

    def bestMatchSentence(self):
        scores = self.match_sentence()

        # Create a list of tuples sorted by index 1 i.e. value field     
        listofTuples = sorted(scores.items() ,  key=lambda x: x[1], reverse = True)

        # Iterate over the sorted sequence
        # for elem in listofTuples :
        #     print(elem[0] , ": " , elem[1] )

        input_sent = listofTuples[0][0]
        print("Best Match Sentence: ", input_sent)
        return input_sent
        
    ######################## returning what, who answers #######################

    def generate_verb_phrase1(self, vb):
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
        for i in range(len(result)):
            if verb_idx - result_idx[i] != verb_result_idx - i:
                delete_list.append(i)
        if len(delete_list) > 0 and result[0] != vb:
            if verb_result_idx > max(delete_list):
                maxi = max(delete_list)
                result = result[maxi+1:]
        return result

    def generate_subject_phrase1(self, vb, vp):
        sub_list = []
        if vb.head.pos == VERB:
            for word in vb.subtree:
                
                if word == vp[0] and (word==vb or word.head==vb): break
                if word.head == vb and word not in vp and word.pos not in [NOUN, PROPN]: continue
                sub_list.append(word)
        else:
            for word in vb.head.subtree:
                if word == vp[0] and (word==vb or word.head==vb): break
                if word.head == vb and word not in vp and word.pos not in [NOUN, PROPN]: continue
                sub_list.append(word)
        return sub_list


    def sub_phrase_answering(self, sent):
        cur_sent = self.nlp(sent)
        verbs = [token for token in cur_sent if token.pos == VERB or token.pos == AUX]

        for verb in verbs:
            subject_list = []
            object_list = []
            vp_list = self.generate_verb_phrase1(verb)
            subject_list = self.generate_subject_phrase1(verb, vp_list)
        return subject_list


    def whoWhat_Answers(self, sent):
        SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
        doc=self.nlp(sent)
        subject_toks_candidate1 = [tok for tok in doc if (tok.dep_ in SUBJECTS) ]
        subject_toks_candidate2 = self.sub_phrase_answering(sent)
        if len(subject_toks_candidate1) >= len(subject_toks_candidate2):
            return subject_toks_candidate1
        else:
            return subject_toks_candidate2

    ######################## returning binary answers ##########################

    def binary_Answers(self, sent, question):
        sent = self.nlp(sent)
        question = self.nlp(question)
        if question[0].text in ["Did", "Do", "Does"]:
            count = 0
            missed_list = []
            question = question[1:]
        sent_lemmatized = [tok.lemma_ for tok in sent]
        for i in question:
            if i.lemma_ not in sent_lemmatized:
                missed_list.append(i)
                count += 1
        if count > 3:
            return "No"
        elif count == 0:
            return "Yes"
        else:
            for i in missed_list:
                if i.pos == AUX or i.pos == DET or i.pos == PRON or i.is_punct:
                    count -= 1
                else:
                    if i.dep_ == "neg":return "No"       
            if count >= 1: 
                return "No"
            else: return "Yes"


    def getAnswer(self):
        input_sent = self.bestMatchSentence()
        result = []
        if self.question.split()[0] in ["Did", "Do", "Does"]:
            result = self.binary_Answers(input_sent, self.question)
        if self.question.split()[0] in ["Who", "What"]:
            result = self.whoWhat_Answers(input_sent)
        if self.question.split()[0] in ["When", "Where"]:
            # Isabel's funciton for when where answering
            pass
        if self.question.split()[0] in ["Why"]:
            # Youce's function for why answering
            pass
        print("Answer: ", result)
        return result





########################### use the Answering object ###########################

question1 = "Who was able to obtain wealth and stability under Ramesses' rule of over half a century?"
question2 = "Did Egyptian armies fought Hittite armies for control of modern-day Syria?"
print("Question: ", question1)
r = open(file = 'Development_data/set1/a3.txt', mode = "r")
text = r.read()

a = Answering(question1, text)
a.getAnswer()

