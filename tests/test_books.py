import json
import spacy
import pytest
import ftfy

from readability_scores import ReadabilityScorer
from .test_data import *


@pytest.fixture(scope="module")
def nlp():
    pipeline = spacy.load("en_core_web_lg")
    pipeline.add_pipe("readability")
    return pipeline


def validate_book(nlp, book):
    doc = nlp(book["text"])
    assert doc._.flesch_kincaid_grade_level == pytest.approx(book["fk_grade"], rel=1e-2)
    assert doc._.flesch_kincaid_reading_ease == pytest.approx(book["fk_ease"], rel=1e-2)
    assert doc._.coleman_liau_index == pytest.approx(book["coleman_liau"], rel=1e-2)
    assert doc._.automated_readability_index == pytest.approx(book["ari"], rel=1e-2)
    assert doc._.smog == pytest.approx(book["smog"], rel=1e-2)
    assert doc._.dale_chall == pytest.approx(book["dale_chall"], rel=1e-2)
    assert doc._.forcast == pytest.approx(book["forcast"], rel=1e-2)


def test_peter_rabbit(nlp):
    with open("tests/samples/peter_rabbit.json") as fp:
        data = json.load(fp)
    validate_book(nlp, data)


def test_tale_two_cities(nlp):
    with open("tests/samples/tale_of_two_cities.json") as fp:
        data = json.load(fp)
    validate_book(nlp, data)


@pytest.mark.parametrize(
    "text,expected",
    [
        (oliver_twist, 11.30),
        (secret_garden, 6.20),
        (flatland, 14.21),
        (grade_1, -0.38),
        (grade_2, 4.67),
        (grade_3, 3.71),
        (grade_4, 3.20),
        (grade_6, 3.53),
        (grade_8, 6.35),
        (grade_10, 7.74),
        (grade_12, 7.87),
        (grade_14, 11.78),
    ],
)
def test_flesch_kincaid_grade_level(text, expected, nlp):
    text = ftfy.fix_text(text)
    text = " ".join(text.split())
    doc = nlp(text)
    assert pytest.approx(expected, rel=1e-2) == doc._.flesch_kincaid_grade_level


@pytest.mark.parametrize(
    "text,expected",
    [
        (oliver_twist, 60.15),
        (secret_garden, 77.47),
        (flatland, 52.64),
        (grade_1, 106.98),
        (grade_2, 80.13),
        (grade_3, 85.72),
        (grade_4, 89.06),
        (grade_6, 92.02),
        (grade_8, 81.91),
        (grade_10, 69.60),
        (grade_12, 63.41),
        (grade_14, 53.13),
    ],
)
def test_flesch_kincaid_reading_ease(text, expected, nlp):
    text = ftfy.fix_text(text)
    text = " ".join(text.split())
    doc = nlp(text)
    assert pytest.approx(expected, rel=1e-2) == doc._.flesch_kincaid_reading_ease


@pytest.mark.parametrize(
    "text,expected", [(oliver_twist, 10.52), (secret_garden, 8.09), (flatland, 9.50)]
)
def test_dale_chall(text, expected, nlp):
    text = ftfy.fix_text(text)
    text = " ".join(text.split())
    doc = nlp(text)
    assert pytest.approx(expected, rel=1e-2) == doc._.dale_chall


@pytest.mark.parametrize(
    "text,expected", [(oliver_twist, 15.81), (secret_garden, 10.93), (flatland, 0)]
)
def test_smog(text, expected, nlp):
    text = ftfy.fix_text(text)
    text = " ".join(text.split())
    doc = nlp(text)
    assert pytest.approx(expected, rel=1e-2) == doc._.smog


@pytest.mark.parametrize(
    "text,expected", [(oliver_twist, 8.45), (secret_garden, 6.38), (flatland, 7.89)]
)
def test_coleman_liau(text, expected, nlp):
    text = ftfy.fix_text(text)
    text = " ".join(text.split())
    doc = nlp(text)
    assert pytest.approx(expected, rel=1e-2) == doc._.coleman_liau_index


@pytest.mark.parametrize(
    "text,expected", [(oliver_twist, 12.02), (secret_garden, 5.45), (flatland, 14.97)]
)
def test_ari(text, expected, nlp):
    text = ftfy.fix_text(text)
    text = " ".join(text.split())
    doc = nlp(text)
    assert pytest.approx(expected, rel=1e-2) == doc._.automated_readability_index


@pytest.mark.parametrize(
    "text,expected", [(oliver_twist, 11.5), (secret_garden, 10.2), (flatland, 12.7)]
)
def test_forcast(text, expected, nlp):
    text = ftfy.fix_text(text)
    text = " ".join(text.split())
    doc = nlp(text)
    assert pytest.approx(expected, rel=1e-2) == doc._.forcast
