class NltkSentenceSplitter(object):
    def __init__(self):
        from nltk import sent_tokenize
        self._sent_tokenize = sent_tokenize

    def __call__(self, text):
        return self._sent_tokenize(text)


class BlingfireSentenceSplitter(object):
    def __init__(self):
        from blingfire import text_to_sentences
        self._text_to_sentences = text_to_sentences

    def __call__(self, text):
        return self._text_to_sentences(text).split('\n')


class SpacySentenceSplitter(object):
    def __init__(self, model_name='en_core_web_sm'):
        import spacy
        self.nlp = spacy.load(model_name)

    def __call__(self, text):
        doc = self.nlp(text)
        return [sent.text for sent in doc.sents]
