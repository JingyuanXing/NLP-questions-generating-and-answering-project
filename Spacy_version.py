import spacy

nlp = spacy.load("en_core_web_sm")
r = open(file = "Development_data/set1/a1.txt", mode = "r")
text = r.read()

paragraphs = text.split('\n')
current_para = paragraphs[3]
doc = nlp(current_para)
sentences = [sent.string.strip() for sent in doc.sents]
print(sentences)

cur_sent = nlp(sentences[0])
for token in cur_sent:
    print(token.text,token.pos_, token.tag_, token.dep_, 
          token.head.text, token.head.pos_)

spacy.displacy.serve(cur_sent, style = "dep") # dep: dependency relationship