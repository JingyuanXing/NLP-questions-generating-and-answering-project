import spacy

questions = ['What is ?', 
'what does the period from the Third Dynasty through to the Sixth Dynasty is ?', 
'what organization period from the Third Dynasty through to the Sixth Dynasty is ?', 
'the period from the what organization Dynasty through to the Sixth Dynasty is ?', 
'the period from the Third what organization through to the Sixth Dynasty is ?', 
'the period from the Third Dynasty through to the what organization Dynasty is ?', 
'the period from the Third what organization through to the Sixth Dynasty is ?', 
'What is regarded as the period from the Third Dynasty through to the Sixth Dynasty ?', 
'what does The Old Kingdom is regarded as ?', 
'The Old Kingdom is regarded as what organization period from the Third Dynasty through to the Sixth Dynasty ?', 
'The Old Kingdom is regarded as the period from the what organization Dynasty through to the Sixth Dynasty ?', 
'The Old Kingdom is regarded as the period from the Third what organization through to the Sixth Dynasty ?', 
'The Old Kingdom is regarded as the period from the Third Dynasty through to the what organization Dynasty ?', 
'The Old Kingdom is regarded as the period from the Third what organization through to the Sixth Dynasty ?'] 

r = open(file = "Development_data/set1/a1.txt", mode = "r")
text = r.read()
paragraphs = text.split('\n')
current_para = paragraphs[3]

nlp = spacy.load("en_core_web_sm")
doc = nlp(current_para)


### POS tagging
for token in doc:
    print(token.text, token.lemma_, token.pos_, token.tag_)
print("\n")

for ques in questions:
    q = nlp(ques)
    for token in q:
        print(token.text, token.lemma_, token.pos_, token.tag_)
print("\n")


### NER tagging
for ent in doc.ents:
    print(ent.text, ent.start_char, ent.end_char, ent.label_)
print("\n")

for ques in questions:
    q = nlp(ques)
    for ent in q.ents:
        print(ent.text, ent.start_char, ent.end_char, ent.label_)
print("\n")









