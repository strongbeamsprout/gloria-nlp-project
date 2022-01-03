import random
import spacy
import copy
import scispacy
from scispacy.abbreviation import AbbreviationDetector
from scispacy.linking import EntityLinker


class MaskAll:
    def __init__(self, mask_token):
        self.mask_token = mask_token

    def __call__(self, text):
        return self.mask_token


class TextMasker:
    def __init__(self, mask_token, mask_prob=.15):
        self.mask_token = mask_token
        self.mask_prob = mask_prob


class WordMasker(TextMasker):
    def __call__(self, text):
        return mask_words(text, self.mask_token, mask_prob=self.mask_prob)


class SentenceMasker(TextMasker):
    def __init__(self, mask_token, mask_prob=.5, nlp=None):
        super().__init__(mask_token, mask_prob=mask_prob)
        self.nlp = nlp if nlp is not None else spacy.load('en_core_web_sm')

    def __call__(self, text):
        return mask_sentences(
            text, self.mask_token, self.nlp, mask_prob=self.mask_prob)


class EntityMasker(TextMasker):
    def __init__(self, mask_token, mask_prob=.5, nlp=None, trim_entities_func=None):
        super().__init__(mask_token, mask_prob=mask_prob)
        self.nlp = nlp if nlp is not None else spacy.load('en_core_web_sm')
        self.trim_entities_func = trim_entities_func

    def __call__(self, text):
        return mask_entities(
            text, self.mask_token, self.nlp, mask_prob=self.mask_prob,
            trim_entities_func=self.trim_entities_func)


class ClinicalEntityMasker(EntityMasker):
    def __init__(self, mask_token, mask_prob=.5):
        scinlp = spacy.load('en_core_sci_sm')
        scinlp.add_pipe("abbreviation_detector")
        scinlp.add_pipe("scispacy_linker", config={"resolve_abbreviations": True, "linker_name": "umls"})
        def trim_entities_func(entities):
            # TODO: implement this
            return entities
        super().__init__(mask_token, mask_prob=mask_prob, nlp=scinlp, trim_entities_func=trim_entities_func)


def mask_words(text, mask_token, mask_prob=.15):
    words = text.split()
    num_masks = max(int(len(words) * mask_prob), 1)
    num_masks = min(num_masks, max(len(words) - 1, 0))
    indices = list(range(len(words)))
    random.shuffle(indices)
    for i in indices[:num_masks]:
        words[i] = mask_token
    return ' '.join(words)


def mask_entities(text, mask_token, nlp, mask_prob=.15, trim_entities_func=None):
    entities = list(nlp(text).ents)
    if trim_entities_func is not None:
        entities = trim_entities_func(entities)
    num_masks = max(int(len(entities) * mask_prob), 1)
    num_masks = min(num_masks, max(len(entities) - 1, 0))
    indices = list(range(len(entities)))
    random.shuffle(indices)
    entities_to_mask = sorted([entities[i] for i in indices[:num_masks]], key=lambda x: x.start_char)
    segments = []
    offset = 0
    for ent in entities_to_mask:
        segments.append(text[offset:ent.start_char])
        segments.append(mask_token)
        offset = ent.end_char
    segments.append(text[offset:])
    return ''.join(segments)


def mask_sentences(text, mask_token, nlp, mask_prob=.15):
    sentences = list(nlp(text).sents)
    num_masks = max(int(len(sentences) * mask_prob), 1)
    num_masks = min(num_masks, max(len(sentences) - 1, 0))
    indices = list(range(len(sentences)))
    random.shuffle(indices)
    sentences_to_mask = sorted([sentences[i] for i in indices[:num_masks]], key=lambda x: x.start_char)
    segments = []
    offset = 0
    for sent in sentences_to_mask:
        segments.append(text[offset:sent.start_char])
        segments.append(mask_token)
        offset = sent.end_char
    segments.append(text[offset:])
    return ''.join(segments)
