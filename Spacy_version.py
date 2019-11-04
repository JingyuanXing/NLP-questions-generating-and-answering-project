import spacy

nlp = spacy.load("en_core_web_sm")
r = open(file = "set1/set1/a1.txt", mode = "r")
text = r.read()

###

paragraphs = text.split('\n')
current_para = paragraphs[3]
doc = nlp(current_para)
sentences = [sent.string.strip() for sent in doc.sents]
print(sentences)