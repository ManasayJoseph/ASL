# from spellchecker import SpellChecker

# spell = SpellChecker()

# # find those words that may be misspelled
# misspelled = spell.unknown(['bodu', 'is', 'hapenning', 'here'])

# for word in misspelled:
#     # Get the one `most likely` answer
#     print(spell.correction(word))

#     # Get a list of `likely` options
#     # print(spell.candidates(word))

import spacy

nlp = spacy.load("en_core_web_sm")

def word_recommendation(text):
    doc = nlp(text)

    recommendations = []
    for token in doc:
        if token.is_alpha and not token.is_stop:
            suggestions = [word.orth_ for word in token._.holmes.suggest()]
            recommendations.append((token.text, suggestions))

    return recommendations

word_recommendation("happeni")