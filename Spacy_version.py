import spacy

nlp = spacy.load("en_core_web_sm")
r = open(file = "Development_data/set1/a1.txt", mode = "r")
text = r.read()

###

paragraphs = text.split('\n')
current_para = paragraphs[3]
doc = nlp(current_para)
sentences = [sent.string.strip() for sent in doc.sents]
#print(sentences)

###

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
    for i in delete_list:
        result.pop(i)
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

###

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
                subject_list = generate_subject_phrase(verb, vp_list, self.relationships)
                object_list = generate_object_phrase(verb, vp_list)
                if len(object_list) > 0 and len(subject_list) > 0:
                    relationship = (subject_list, vp_list, object_list)
                    self.relationships.append(relationship)
        return self.relationships


###

a = NLPparagraph(current_para)
print(a.sent_relation_extraction())










