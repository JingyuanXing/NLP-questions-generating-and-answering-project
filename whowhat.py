import spacy
from spacy.symbols import nsubj, VERB, AUX, NOUN, attr, dobj, agent, amod, prep, advmod, auxpass
nlp = spacy.load("en_core_web_sm")



ques_what = "What is most commonly regarded as the period from the Third Dynasty through to the Sixth Dynasty (2686–2181 BC)?"
ques_who = "Who commissioned the building of not one, but three pyramids?"
ques_who_middle = "Who learned to express their culture's worldview?"

sent_what = "The Old Kingdom is most commonly regarded as the period from the Third Dynasty through to the Sixth Dynasty (2686–2181 BC)."
sent_who = "Snefru commissioned the building of not one, but three pyramids."
sent_who_middle = "During this period, artists learned to express their culture's worldview."

####


#given a verb, generate the verb phrase around the verb
def generate_verb_phrase1(vb):
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
        if verb_result_idx > max(delete_list):
            maxi = max(delete_list)
            result = result[maxi+1:]
    return result

#given the verb phrase and the verb and the main relations, try to find the corresponding subject part of the relation
def generate_subject_phrase1(vb, vp):
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

def sub_phrase_answering(sent):
    cur_sent = nlp(sent)
    verbs = [token for token in cur_sent if token.pos == VERB or token.pos == AUX]

    for verb in verbs:
        subject_list = []
        object_list = []
        vp_list = generate_verb_phrase1(verb)
        subject_list = generate_subject_phrase1(verb, vp_list)
    return subject_list

###

def findAnswer(sent):
    SUBJECTS = ["nsubj", "nsubjpass", "csubj", "csubjpass", "agent", "expl"]
    doc=nlp(sent)
    subject_toks_candidate1 = [tok for tok in doc if (tok.dep_ in SUBJECTS) ]
    subject_toks_candidate2 = sub_phrase_answering(sent)
    if len(subject_toks_candidate1) >= len(subject_toks_candidate2):
        return subject_toks_candidate1
    else:
        return subject_toks_candidate2


print(findAnswer(sent_what))
print(findAnswer(sent_who))
print(findAnswer(sent_who_middle))