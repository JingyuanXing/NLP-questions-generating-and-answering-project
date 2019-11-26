
"""
Group No Logic Please

Code for answer generation, will be compiled into executables

"""

import warnings
warnings.filterwarnings("ignore")

import spacy
from spacy.symbols import nsubj, VERB, AUX, NOUN, attr, dobj, agent, amod, prep, advmod, auxpass, PROPN, DET, PRON, ADJ, ADV
from spacy.matcher import Matcher
import sys

from collections import Counter
# Called by sentenceCheck
# INPUT: a spacy question; OUTPUT: a spacy sentence
nlp = spacy.load("en_core_web_lg")


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







class Answering(object):
    def __init__(self, questions, text):
        self.questions = questions
        self.text = text
        self.nlp = spacy.load("en_core_web_lg")
        self.nlp_md = spacy.load("en_core_web_lg")
        self.text_nlp = self.nlp(text)
        self.textsplit = [sent.text.strip() for sent in self.text_nlp.sents]
        ref_section = ['Bibliography', 'References']
        idx = len(self.textsplit)
        for i in ref_section:
            if i in self.textsplit:
                idx = self.textsplit.index(i)
            break
        self.textsplit = self.textsplit[:idx]


    ######################## returning selected sentence #######################

    def match_sentence(self, question):
        matcher = Matcher(self.nlp.vocab)
        sent_tokens = [self.nlp(i) for i in self.textsplit]
        question_nlp = self.nlp(question)
        # for t in q_token_set:
        #     question_pattern.extend([{'LEMMA': t, 'OP': '?'}])
        matched_scores = {}

        for sent_token in sent_tokens:
            sent_pattern = []
            this_length = 0
            sent_token_set = set([tok.lemma_.lower() for tok in sent_token])
            for t in sent_token_set:
                sent_pattern.extend([{'LEMMA': t, 'OP': '?'}])
            matcher.add("sent_pattern", None, sent_pattern)
            matches = matcher(question_nlp)
            for match_id, start, end in matches:
                # string_id = nlp.vocab.strings[match_id]  # Get string representation
                # span = question_nlp[start:end]  # The matched span
                # this_length += len(span.text.split(" "))
                this_length += end - start
            matched_scores[sent_token] = this_length
            matcher.remove("sent_pattern")

        matched_scores = {k: v/len(question_nlp) for k, v in matched_scores.items()}

        # from the default similarity function in spacy, find the simiarity score
        # for the sentence, and then weight it with the matching score.
        final_scores = {}

        if question_nlp[0].text in ["Did", "Do", "Does", "Is", "Are", "Were", "Was", "Had", "Has", "Have"]:
            for sent in matched_scores.keys():
                if len(sent) <= 2: continue
                similarity_score = sent.similarity(question_nlp)
                # print(sent)
                # print("match_score: %.3f" % matched_scores[sent])
                # print("similarity score: %.3f" % similarity_score)
                final_scores[sent.text] = 0.7*matched_scores[sent] + 0.3*similarity_score
        else:
            for sent in matched_scores:
                if len(sent) <= 2: continue
                similarity_score = sent.similarity(question_nlp)
                # print(sent)
                # print(matched_scores[sent])
                final_scores[sent.text] = 0.5*matched_scores[sent] + 0.5*similarity_score
        return final_scores

    def bestMatchSentence(self, quest):
        scores = self.match_sentence(quest)
        # Create a list of tuples sorted by index 1 i.e. value field
        listofTuples = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        # Iterate over the sorted sequence
        # for elem in listofTuples :
        #     print(elem[0] , ": " , elem[1] )
        # print(listofTuples[:10])
        input_sent = listofTuples[0][0]
        score = listofTuples[0][1]
        # print("Best Match Sentence: %.3f" % score)
        # print(input_sent)
        return input_sent, score

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
        sent_relevant = [x.strip() for x in sent.split(',')]
        sent_relevant = sent_relevant[0]
        doc = self.nlp(sent_relevant)
        ques = self.nlp(question)
        # print(sent)

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


    ##############################################################################3

    def whenWhereQues(self, whereQues, ansSentence):
        question = self.nlp(whereQues)
        sentence = self.nlp(ansSentence)
        NERlst = []
        questionVerb = None
        questionSubject = None
        questionWord = "where"
        for word in question:
            if word.text in ["where", "Where"]:
                questionWord = "where"
            elif word.text in ["When", "when"]:
                questionWord = "when"
            elif word.pos_ in ["VERB", "AUX"]:
                questionVerb = word
            elif word.dep_ in ["nsubj", "nsubjpass"] and questionSubject is None:
                questionSubject = word.text
        for tok in sentence.ents:
            if questionWord == "where" and tok.label_ == "GPE":
                NERlst.append(tok.text)
            elif questionWord == "when" and tok.label_ == "DATE":
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
            # elif word.dep_ in ["nsubj", "nsubjpass"] and (word.text.lower() == questionSubject.lower() or (word.pos_ == "PRON" and subject == [])):
            #    subject.append(word.text)
            elif questionVerb is not None and word.pos_ == "VERB" and word.dep_ in ["auxpass", "ROOT"] and word.head.lemma_ == questionVerb.lemma_:
                verb.append(word.text)
            elif word.pos_ == "VERB" and word.lemma_ == questionVerb.lemma_:
                verb.append(word.text)
            # elif word.pos_ in ["DET", "ADJ"] and word.head.text == questionSubject.lower():
            # ß    subject.append(word.text)
            # elif word.pos_ == "ADP" and word.head.text in verb:
            #    prep = word.text
            #    prepPhrase.append(prep)
            # elif word.head.text == prep:
            #    if word.pos_ in ["NN", "NNS"]:
            #        prepPhrase.append("the")
            #        prepPhrase.append(word.text)
            #    else:
            #        prepPhrase.append(word.text)
            elif word.dep_ in ["nummod"] and questionWord == "when":
                for w in word.subtree:
                    prepPhrase.append(w.text)
        if prepPhrase == [] and NERlst == []:
            return sentence.text

        question = question[1:]  # get rid of "where" or "when"
        if question[0].pos_ == "VERB":
            question = question[1:]
        if questionSubject is None: return sentence.text
        elif verb == [] and prepPhrase == [] and questionVerb is not None and len(NERlst) > 0:
            ans = questionSubject + " " + questionVerb.text + " " + " ".join(NERlst) + "."
        elif verb == [] and questionVerb is not None and len(prepPhrase) > 0:
            ans = questionSubject + " " + questionVerb.text + " " + " ".join(prepPhrase) + "."
        elif prepPhrase == [] and verb is not None and len(NERlst) > 0:
            ans = questionSubject + " " + " ".join(verb) + " " + " ".join(NERlst) + "."
        else:
            return sentence.text
        return ans


    ######################## returning binary answers ##########################

    def binary_Answers(self, sent, question):
        sent = self.nlp(sent)
        question = self.nlp(question)
        missed_list = []
        count = 0
        neg = 1
        question = question[1:]
        sent_lemmatized = [tok.lemma_.lower() for tok in sent]
        for i in question:
            if i.lemma_.lower() not in sent_lemmatized:
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
                    if i.dep_ == "neg":
                        count -= 1
                        neg = neg * -1
            if count > 1 or neg < 0:
                return "No"
            else:
                return "Yes"


    ######################## find why#####################


    def Why_Answer(self, orig_sent, question):
        sent = self.nlp(orig_sent)
        if len(sent) <= 3: return sent
        question = self.nlp(question)
        sent_token_list = [tok.text.lower() for tok in sent]
        this_sent = False
        for sent_tk in sent_token_list:
            if sent_tk in ['because', 'since', 'due']:
                this_sent = True
        beginning = " ".join([i.text.lower() for i in sent[:3]])
        ReasonBefore = True if beginning in ['for this reason', 'as a result'] else False
        cur_idx = self.textsplit.index(orig_sent)
        if sent[0].text in ["Therefore", "Thus", "Consequently", "Hence"] or ReasonBefore:
            if cur_idx > 0:
                return self.textsplit[cur_idx-1]
            else:
                return sent.text
        if cur_idx < len(self.textsplit) - 1 and not this_sent:
            nextline = self.nlp(self.textsplit[cur_idx+1])
            for ind, tok in enumerate(nextline):
                if tok.text.lower() in ['because', 'since', 'due']:
                    res = [tk.text for tk in tok.head.subtree]
                    return " ".join(res)
        root = [token for token in question if token.head == token]
        sent_verbs = [tok for tok in sent if tok.pos == AUX or tok.pos == VERB]
        if len(sent_verbs) == 0: return sent.text
        root_verb_toks = root + sent_verbs
        root_new = root_verb_toks[0]
        similarity_verb = [(root_new.similarity(tok), tok, idx) for idx, tok in enumerate(root_verb_toks[1:])]
        sorted_sim_verb = sorted(similarity_verb, key=lambda x: x[0], reverse=True)
        if sorted_sim_verb[0][0] > 0.6:
            for sim, tk, idx in sorted_sim_verb:
                for ci in tk.children:
                    if ci.text.lower() in ["since", "because", "based", "due"]:
                        sbtree = [i for i in ci.subtree]
                        if len(sbtree) <= 2:
                            res = [i.text for i in tk.subtree]
                        else:
                            res = [i.text for i in ci.subtree]
                        return " ".join(res)
                    for cci in ci.children:
                        if cci.text.lower() in ["since", "because", "based", "due"]:
                            sbtree = [i for i in cci.subtree]
                            if len(sbtree) <= 2:
                                res = [i.text for i in ci.subtree]
                                return " ".join(res)
                            else:
                                res = [i.text for i in cci.subtree]
                                return " ".join(res)
                    if this_sent: continue
                    if ci.text.lower() in ["for", "as", "to"]:
                        res = [i.text for i in ci.head.subtree]
                        return " ".join(res)
            return sent.text
        else:
            for c in root[0].children:
                if c.text.lower() in ["since", "because", "based", "due"]:
                    res = [i.text for i in c.subtree]
                    return " ".join(res)
                if c.pos != VERB or c.pos != NOUN:
                    continue
                for cc in c.children:
                    if cc.text.lower() in ['since', 'because', 'due', 'based']:
                        sbtree = [i.text for i in cc.subtree]
                        if len(sbtree) <= 2:
                            res = [i.text for i in c.subtree]
                            return " ".join(res)
                        else:
                            res = [i.text for i in cc.subtree]
                            return " ".join(res)
                if this_sent: continue
                if c.text.lower in ['for', 'as', 'to']:
                    res = [i.text for i in c.subtree]
                    return " ".join(res)
            return sent.text


    #####################################################

    def HowAnswer(self, sent, question):
        sent_nlp = self.nlp(sent)
        ques_nlp = self.nlp(question)
        qroot_verb = [i for i in ques_nlp if i.head == i][0]
        s_verbs = [i for i in sent_nlp if i.pos in [VERB, AUX]]
        if ques_nlp[1].pos != ADJ and ques_nlp[1].pos != ADV:
            if len(s_verbs) == 0: return sent.text
            similarity_list = [(tk, qroot_verb.similarity(tk)) for tk in s_verbs]
            similarity_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)
            for tk, sim in similarity_list:
                res = []
                for c in tk.children:
                    if c.dep == prep:
                        res += [i for i in c.subtree]
                        if res[-1].pos_ == "CCONJ": res.pop(-1)
                if len(res) > 0:
                    res = [i.text for i in res]
                    return " ".join(res)
            return sent_nlp.text
        else:
            if qroot_verb.pos == AUX:
                return sent_nlp.text
            else:
                similarity_list = [(tk, qroot_verb.similarity(tk)) for tk in s_verbs]
                similarity_list = sorted(similarity_list, key=lambda x: x[1], reverse=True)
                for tk, sim in similarity_list:
                    res = [i.text for i in tk.subtree]
                    if len(res) > 2:
                        return " ".join(res)
                return sent_nlp.text




    ######################## find cor #####################

    def getAnswer(self):
        result = []
        for ques in self.questions:
            q_list_dummy = [self.nlp(ques)]
            score_list_dummy = sentenceCheck(q_list_dummy, True)
            input_sent, score = self.bestMatchSentence(ques)
            if score_list_dummy[0] == 0 or score < 0.25 or len(self.nlp(ques)) <= 2:
                result.append("This question is too problematic to answer")
                continue
            if ques.split()[0] in ["Did", "Do", "Does", "Is", "Are", "Were", "Was", "Had", "Has", "Have", "Must", "Could", "Can", "Would"]:
                result.append(self.binary_Answers(input_sent, ques))
            elif ques.split()[0] in ["Who", "What", "Which", "Whom", "Whose"]:
                result.append(self.whoWhat_Answers(input_sent, ques))
            elif ques.split()[0] in ["When", "Where"]:
                result.append(self.whenWhereQues(ques, input_sent))
            elif ques.split()[0] in ["Why"]:
                result.append(self.Why_Answer(input_sent, ques))
            elif ques.split()[0] in ['How']:
                result.append(self.HowAnswer(input_sent, ques))
            else:
                result.append(input_sent)
        return result

#
# question1 = "Who was able to obtain wealth and stability under Ramesses' rule of over half a century?"
# question2 = "Did Egyptian armies fought Hittite armies for control of modern-day Syria?"

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
        print("A" + str(i+1) + ". " + ans)
    return


if __name__ == "__main__":
    main()
