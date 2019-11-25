#!/usr/bin/env python3

"""
Group No Logic Please

Code for answer generation, will be compiled into executables

"""


import spacy
from spacy.symbols import nsubj, VERB, AUX, NOUN, attr, dobj, agent, amod, prep, advmod, auxpass, PROPN, DET, PRON
from spacy.matcher import Matcher

import sys

from collections import Counter



class Answering(object):
    def __init__(self, questions, text):
        self.questions = questions
        self.text = text
        self.textsplit = self.text.splitlines()
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp_md = spacy.load("en_core_web_md")

    ######################## returning selected sentence #######################

    def match_sentence(self, question):
        matcher = Matcher(self.nlp.vocab)
        text_raw = self.nlp_md(self.text)
        sent_tokens = [sent.string.strip() for sent in text_raw.sents]

        question_tokens = question.split(" ")
        question_pattern = [];
        for t in question_tokens:
            question_pattern.extend([{'LEMMA': t, 'OP': '?'}])

        matcher.add("question_pattern", None, question_pattern)

        matched_scores = {}
        max_length = 0

        for sent_token in sent_tokens:
            this_length = 0
            this_score = 0
            doc = self.nlp_md(sent_token)
            matches = matcher(doc)
            for match_id, start, end in matches:
                # string_id = nlp.vocab.strings[match_id]  # Get string representation
                span = doc[start:end]  # The matched span
                this_length += len(span.text.split(" "))
            max_length = max(this_length, max_length)
            matched_scores[sent_token] = this_length

        matched_scores = {k: (v / max_length) for k, v in matched_scores.items()}

        # from the default similarity function in spacy, find the simiarity score
        # for the sentence, and then weight it with the matching score.
        final_scores = {}

        question_nlp = self.nlp_md(question)
        for sent in matched_scores:
            if len(self.nlp(sent)) <= 1: continue
            similarity_score = self.nlp_md(sent).similarity(question_nlp)
            # print(sent)
            # print(matched_scores[sent])
            final_scores[sent] = (matched_scores[sent] + similarity_score) / 2

        return (final_scores)

    def bestMatchSentence(self, quest):
        scores = self.match_sentence(quest)

        # Create a list of tuples sorted by index 1 i.e. value field
        listofTuples = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Iterate over the sorted sequence
        # for elem in listofTuples :
        #     print(elem[0] , ": " , elem[1] )

        input_sent = listofTuples[0][0]
        #print("Best Match Sentence: ", input_sent)
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
                    verb_result_idx = len(result) - 1
        for i in range(len(result)):
            if verb_idx - result_idx[i] != verb_result_idx - i:
                delete_list.append(i)
        if len(delete_list) > 0 and result[0] != vb:
            if verb_result_idx > max(delete_list):
                maxi = max(delete_list)
                result = result[maxi + 1:]
        return result

    def generate_subject_phrase1(self, vb, vp):
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

    def sub_phrase_answering(self, sent):
        cur_sent = self.nlp(sent)
        verbs = [token for token in cur_sent if token.pos == VERB or token.pos == AUX]

        for verb in verbs:
            subject_list = []
            vp_list = self.generate_verb_phrase1(verb)
            subject_list = self.generate_subject_phrase1(verb, vp_list)
        return subject_list

    def generate_object_phrase1(self, vb, vp):
        ob_list = []
        okay_to_insert_object = False
        for ind, word in enumerate(vb.subtree):
            if word in vp and (word.head == vb or word == vb):
                okay_to_insert_object = True
            elif word not in vp and okay_to_insert_object:
                ob_list.append(word)
        return ob_list

    def obj_phrase_answering(self, sent):
        cur_sent = self.nlp(sent)
        verbs = [token for token in cur_sent if token.pos == VERB or token.pos == AUX]

        for verb in verbs:
            object_list = []
            vp_list = self.generate_verb_phrase1(verb)
            object_list = self.generate_object_phrase1(verb, vp_list)
        return object_list


    def whoWhat_Answers(self, sent, question):
        SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
        OBJECTS = ["dobj", "dative", "attr", "oprd"]
        print(sent, "\n")
        sent_relevant = [x.strip() for x in sent.split(',')]
        sent_relevant = sent_relevant[0]
        doc = self.nlp(sent_relevant)
        ques = self.nlp(question)
        

        subject_candidate1 = [tok for tok in doc if (tok.dep_ in SUBJECTS)]
        subject_candidate2 = self.sub_phrase_answering(sent)
        object_candidate1 = [tok for tok in doc if (tok.dep_ in OBJECTS)]
        object_candidate2 = self.obj_phrase_answering(sent)
        for word in subject_candidate1:
            if word.text in question and word.text != "the":
                subject_candidate1 = []
        for word in subject_candidate2:
            if word.text in question and word.text != "the":
                subject_candidate2 = []
        for word in object_candidate1:
            if word.text in question and word.text != "the":
                object_candidate1 = []
        for word in object_candidate2:
            if word.text in question and word.text != "the":
                object_candidate2 = []

        subl1 = len(subject_candidate1)
        subl2 = len(subject_candidate2)
        objl1 = len(object_candidate1)
        objl2 = len(object_candidate2)
        if (subl1 == 0 and subl2 == 0 and objl1 == 0 and objl2 == 0): 
            return sent
        else:
            if subl1 == max(subl1, subl2, objl1, objl2):
                return " ".join([k.text for k in subject_candidate1])
            elif subl2 == max(subl1, subl2, objl1, objl2):
                return " ".join([k.text for k in subject_candidate2])
            elif objl1 == max(subl1, subl2, objl1, objl2):
                return " ".join([k.text for k in object_candidate1])
            else:
                return " ".join([k.text for k in object_candidate2])


    ###################### returning when, where answers #######################

    def whenWhereQues(self, whereQues, ansSentence):
        question = self.nlp(whereQues)
        sentence = self.nlp(ansSentence)
        NERlst = []
        questionVerb = None
        questionSubject = None
        for word in question:
            if word.pos_ == "VERB":
                questionVerb = word
            if word.dep_ in ["nsubj", "nsubjpass"] and questionSubject is None:
                questionSubject = word.text.lower()
        for tok in sentence.ents:
            if tok.label_ == "GPE":
                NERlst.append(tok.text)
        prepPhrase = []
        loc = []
        prep = None
        subject = []
        verb = []
        for word in sentence:
            if prep is not None and prepPhrase is not None and (
                    word.pos_ == "CCONJ" or word.tag_ == "CC"):  # going into second clause of the sentence, but we have our answer already
                break
            elif word.dep_ in ["nsubj", "nsubjpass"] and word.text == questionSubject:
                subject.append(word.text)
            elif word.pos_ == "VERB" and word.dep_ == "auxpass" and word.head.lemma_ == questionVerb.lemma_:
                verb.append(word.text)
            elif word.pos_ == "VERB" and word.lemma_ == questionVerb.lemma_:
                verb.append(word.text)
            elif word.pos_ in ["DET", "ADJ"] and word.head.text == questionSubject:
                subject.append(word.text)
            elif word.pos_ == "ADP" and word.head.text in verb:
                prep = word.text
                prepPhrase.append(prep)
            elif word.head.text == prep:
                if word.pos_ in ["NN", "NNS"]:
                    prepPhrase.append("the")
                    prepPhrase.append(word.text)
                else:
                    prepPhrase.append(word.text)

        if subject is None or verb is None or prep is None:
            return sentence.text

        question = question[1:]  # get rid of "where"
        if question[0].pos_ == "VERB":
            question = question[1:]
        ans = " ".join(subject) + " " + " ".join(verb) + " " + " ".join(prepPhrase) + "."
        return ans

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
                    if i.dep_ == "neg": return "No"
            if count >= 1:
                return "No"
            else:
                return "Yes"


    ######################## find why#####################

    def Why_Answer(self, sent, question):
        sent = self.nlp(sent)
        question = self.nlp(question)
        if sent[0].text in ["Therefore", "Thus", "Consequently", "Hence"]:
            cur_idx = self.textsplit.index(sent.text)
            if cur_idx > 0:
                return self.textsplit[cur_idx-1]
            else:
                return sent.text
        root = [token for token in question if token.head == token]
        sent_verbs = [tok for tok in sent if tok.pos == AUX or tok.pos == VERB]
        if len(sent_verbs) == 0: return sent.text
        root_verb_toks = root + sent_verbs
        root_verb_nlp = self.nlp(" ".join([t.text for t in root_verb_toks]))
        root_new = root_verb_nlp[0]
        similarity_verb = [(root_new.similarity(tok), tok, idx) for idx, tok in enumerate(root_verb_nlp[1:])]
        sorted_sim_verb = sorted(similarity_verb, key=lambda x: x[0], reverse=True)
        if sorted_sim_verb[0][0] > 0.6:
            for sim, i, idx in sorted_sim_verb:
                for ci in sent_verbs[idx].children:
                    for cci in ci.children:
                        if cci.text.lower() in ["since", "because", "to"]:
                            res = [i.text for i in ci.subtree]
                            return " ".join(res)
                    if ci.text.lower() in ["for", "as", "due"]:
                        res = [i.text for i in ci.head.subtree]
                        return " ".join(res)
            return sent.text
        else:
            for c in root[0].children:
                if c.pos != VERB:
                    continue
                for cc in c.children:
                    if cc.text.lower() in ['since', 'because', 'to']:
                        res = [i.text for i in c.subtree]
                        return " ".join(res)
                if c.text.lower in ['for', 'as', 'due']:
                    res = [i.text for i in c.subtree]
                    return " ".join(res)
            return sent.text



    ######################## find cor #####################

    def getAnswer(self):
        result = []
        for ques in self.questions:
            input_sent = self.bestMatchSentence(ques)
            print(input_sent)
            if ques.split()[0] in ["Did", "Do", "Does", "Is", "Are", "Were", "Was", "Had", "Has", "Have"]:
                result.append(self.binary_Answers(input_sent, ques))
            elif ques.split()[0] in ["Who", "What", "Which"]:
                result.append(self.whoWhat_Answers(input_sent, ques))
            elif ques.split()[0] in ["When", "Where"]:
                result.append(self.whenWhereQues(ques, input_sent))
            elif ques.split()[0] in ["Why"]:
                result.append(self.Why_Answer(input_sent, ques))
            else:
                result.append(input_sent)
        return result


def main():
    r_path, q_path = sys.argv[1], sys.argv[2]
    q = open(file=q_path, mode="r", encoding="utf-8")
    r = open(file=r_path, mode="r", encoding="utf-8")
    textr = r.read()
    textq = q.read()
    Question_list = textq.splitlines()

    a = Answering(Question_list, textr)
    answers = a.getAnswer()
    for i, ans in enumerate(answers):
        print("Q" + str(i+1) + ". " + ans)
    return


if __name__ == "__main__":
    main()
