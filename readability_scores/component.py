import re
from math import sqrt

import pyphen
import spacy
from spacy.language import Language
from spacy.tokens import Doc, Token

from .constants import DALE_CHALL_WORDS


class ReadabilityScorer:
    """spaCy v2.0 pipeline component for calculating readability scores of of text.
    Provides scores for Flesh-Kincaid grade level, Flesh-Kincaid reading ease, and Dale-Chall.
    USAGE:
        >>> import spacy
        >>> from spacy_readability import Readability
        >>> nlp = spacy.load('en')
        >>> read = Readability()
        >>> nlp.add_pipe(read, last=True)
        >>> doc = nlp("I am some really difficult text. I use obnoxiously large words.")
        >>> print(doc._.flesch_kincaid_grade_level)
        >>> print(doc._.flesch_kincaid_reading_ease)
        >>> print(doc._.dale_chall)
        >>> print(doc._.smog)
        >>> print(doc._.coleman_liau_index)
        >>> print(doc._.automated_readability_index)
        >>> print(doc._.forecast)
    """

    def __init__(self, nlp: Language):
        """
        Initialize the pipeline and add the extensions
        """
        lang = nlp.meta.get("lang")
        self.dic = pyphen.Pyphen(lang=lang)
        if not Doc.has_extension("flesch_kincaid_grade_level"):
            Doc.set_extension("flesch_kincaid_grade_level", getter=self.fk_grade)

        if not Doc.has_extension("flesch_kincaid_reading_ease"):
            Doc.set_extension("flesch_kincaid_reading_ease", getter=self.fk_ease)

        if not Doc.has_extension("dale_chall"):
            Doc.set_extension("dale_chall", getter=self.dale_chall)

        if not Doc.has_extension("smog"):
            Doc.set_extension("smog", getter=self.smog)

        if not Doc.has_extension("coleman_liau_index"):
            Doc.set_extension("coleman_liau_index", getter=self.coleman_liau)

        if not Doc.has_extension("automated_readability_index"):
            Doc.set_extension("automated_readability_index", getter=self.ari)

        if not Doc.has_extension("forcast"):
            Doc.set_extension("forcast", getter=self.forcast)

    def __call__(self, doc: Doc) -> Doc:
        """
        Apply the pipeline to the document.
        """
        return doc

    def _get_num_sentences(self, doc: Doc) -> int:
        """
        Returns the number of sentences in the given document.

        Parameters:
            doc (Doc): The document to count the sentences in.

        Returns:
            int: The number of sentences in the document.
        """
        return len(list(doc.sents))

    def _get_num_words(self, doc: Doc) -> int:
        """
        Calculate the number of words in a given document.

        Parameters:
        - doc (Doc): The document to calculate the number of words for.

        Returns:
        - int: The number of words in the document.
        """
        return len(list(filter(lambda x: not x.is_punct, doc)))

    def _get_num_syllables(self, token: Token) -> int:
        """
        Calculates the number of syllables in a given token.

        Args:
            token (Token): The token for which to calculate the number of syllables.

        Returns:
            int: The number of syllables in the token.
        """
        # Not a word in the spacy vocabulary
        if token.is_oov:
            return 0

        # Try using pyphen
        try:
            count = len(self.dic.inserted(token.text).split("-"))
        except TypeError:
            return 0

        # we found more than one syllable
        if count > 1:
            return count

        # Use Regex looking for vowel groups
        # to handle cases like "amen" where pyphen doesn't hyphenate.
        pattern = r"[aeiouáéíóúãõâêôûàAEIOU]+"
        matches = re.findall(pattern, token.text)
        return len(matches)

    def fk_grade(self, doc):
        """
        Calculates the Flesch-Kincaid Grade Level of a document.

        Parameters:
            doc (str): The document to be analyzed.

        Returns:
            float: The Flesch-Kincaid Grade Level score of the document.
        """

        num_sents = self._get_num_sentences(doc)
        num_words = self._get_num_words(doc)
        num_syllables = sum(self._get_num_syllables(token) for token in doc)

        if num_sents == 0 or num_words == 0 or num_syllables == 0:
            return 0

        return (
            (11.8 * num_syllables / num_words) + (0.39 * num_words / num_sents) - 15.59
        )

    def fk_ease(self, doc):
        """
        Calculates the Flesch-Kincaid Ease score for a given document.

        Parameters:
        - doc: The document to calculate the score for.

        Returns:
        - The Flesch-Kincaid Ease score for the document.
        """
        num_sents = self._get_num_sentences(doc)
        num_words = self._get_num_words(doc)
        num_syllables = sum(self._get_num_syllables(token) for token in doc)

        if num_sents == 0 or num_words == 0 or num_syllables == 0:
            return 0
        words_per_sent = num_words / num_sents
        syllables_per_word = num_syllables / num_words

        return 206.835 - (1.015 * words_per_sent) - (84.6 * syllables_per_word)

    def dale_chall(self, doc):
        """
        Calculate the Dale-Chall readability score for a given document.

        Parameters:
        - doc (spacy.tokens.doc.Doc): The document to calculate the readability score for.

        Returns:
        - float: The Dale-Chall readability score for the document.
        """
        num_sents = self._get_num_sentences(doc)
        num_words = self._get_num_words(doc)
        if num_sents == 0 or num_words == 0:
            return 0

        diff_words_count = 0
        for token in doc:
            if token.lemma_.lower() not in DALE_CHALL_WORDS:
                diff_words_count += 1

        percent_difficult_words = 100 * diff_words_count / num_words
        avg_sentence_length = num_words / num_sents
        grade = 0.1579 * percent_difficult_words + 0.0496 * avg_sentence_length

        if percent_difficult_words > 5:
            grade += 3.6365
        return grade

    def smog(self, doc):
        """
        Returns SMOG score for the document and -1 if the document is shorter
        than 30 sentences.
        """
        num_sents = self._get_num_sentences(doc)
        num_words = self._get_num_words(doc)
        if num_sents < 30 or num_words == 0:
            return 0
        num_poly = sum(
            list(
                filter(
                    lambda x: x >= 3, [self._get_num_syllables(token) for token in doc]
                )
            )
        )
        return 1.0430 * sqrt(num_poly * 30 / num_sents) + 3.1291

    def coleman_liau(self, doc):
        """
        Calculate the Coleman-Liau readability score for a given document.

        Parameters:
        - doc (str): The document to calculate the readability score for.

        Returns:
        - float: The Coleman-Liau readability score.

        Raises:
        - None

        """
        num_words = self._get_num_words(doc)
        if num_words <= 0:
            return 0

        num_sents = self._get_num_sentences(doc)

        letter_count = sum([len(token) for token in doc if token.is_alpha])
        if letter_count <= 0:
            return 0

        letters_per_hundred_word = letter_count / num_words * 100
        sentences_per_hundred_words = num_sents / num_words * 100

        return (
            0.0588 * letters_per_hundred_word
            - 0.296 * sentences_per_hundred_words
            - 15.8
        )

    def ari(self, doc):
        """
        Calculates the Automated Readability Index (ARI) of a given document.

        Parameters:
        - doc (str): The document to calculate the ARI for.

        Returns:
        - float: The ARI score of the document.
        """
        num_sents = self._get_num_sentences(doc)
        num_words = self._get_num_words(doc)

        if num_words <= 0:
            return 0

        letter_count = sum([len(token) for token in doc if not token.is_punct])
        letters_per_word = letter_count / num_words
        words_per_sentence = num_words / num_sents
        return 4.71 * letters_per_word + 0.5 * words_per_sentence - 21.43

    def forcast(self, doc):
        """
        Calculates the forcast readability score for a given document.

        Parameters:
        - doc (list): A list of tokens representing the document.

        Returns:
        - float: The forecast readability score.

        The forecast readability score is calculated based on the number of words in the document.
        If the number of words is less than 150, the score is 0.
        Otherwise, the score is calculated by counting the number of monosyllabic words in the first 150 tokens of the document,
        and subtracting the result divided by 10 from 20.
        """
        num_words = self._get_num_words(doc)

        if num_words < 150:
            return 0

        mono_syllabic = sum(
            list(
                filter(
                    lambda x: x == 1,
                    [self._get_num_syllables(token) for token in doc[:150]],
                )
            )
        )
        return 20 - (mono_syllabic / 10)


@Language.factory("readability")
def create_readability_component(nlp: Language, name: str) -> ReadabilityScorer:
    """
    Create a readability component.

    Args:
        nlp (Language): The spaCy language object.
        name (str): The name of the component.

    Returns:
        ReadabilityScorer: The readability scorer component.
    """
    return ReadabilityScorer(nlp)
