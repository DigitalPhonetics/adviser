############################################################################################
#
# Copyright 2020, University of Stuttgart: Institute for Natural Language Processing (IMS)
#
# This file is part of Adviser.
# Adviser is free software: you can redistribute it and/or modify'
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3.
#
# Adviser is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with Adviser.  If not, see <https://www.gnu.org/licenses/>.
#
############################################################################################

"""This file is derived from https://github.com/keithito/tacotron."""

import re

import inflect
from unidecode import unidecode

_inflect = inflect.engine()
_comma_number_re = re.compile(r'([0-9][0-9,]+[0-9])')
_decimal_number_re = re.compile(r'([0-9]+\.[0-9]+)')
_pounds_re = re.compile(r'£([0-9,]*[0-9]+)')
_dollars_re = re.compile(r'\$([0-9.,]*[0-9]+)')
_ordinal_re = re.compile(r'[0-9]+(st|nd|rd|th)')
_number_re = re.compile(r'[0-9]+')
_ID_number_re = re.compile(r'[0-9]{4,}|0[0-9]+')
_email_re = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-]+\.[a-zA-Z]+")

def _remove_commas(m):
    return m.group(1).replace(',', '')


def _expand_dollars(m):
    match = m.group(1)
    parts = match.split('.')
    if len(parts) > 2:
        return match + ' dollars'  # Unexpected format
    dollars = int(parts[0]) if parts[0] else 0
    cents = int(parts[1]) if len(parts) > 1 and parts[1] else 0
    if dollars and cents:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s, %s %s' % (dollars, dollar_unit, cents, cent_unit)
    elif dollars:
        dollar_unit = 'dollar' if dollars == 1 else 'dollars'
        return '%s %s' % (dollars, dollar_unit)
    elif cents:
        cent_unit = 'cent' if cents == 1 else 'cents'
        return '%s %s' % (cents, cent_unit)
    else:
        return 'zero dollars'


def _expand_ordinal(m):
    return _inflect.number_to_words(m.group(0))


def _expand_number(m):
    num = int(m.group(0))
    if 1000 < num < 3000:
        if num == 2000:
            return 'two thousand'
        elif 2000 < num < 2010:
            return 'two thousand ' + _inflect.number_to_words(num % 100)
        elif num % 100 == 0:
            return _inflect.number_to_words(num // 100) + ' hundred'
        else:
            return _inflect.number_to_words(num, andword='', zero='oh', group=2).replace(', ', ' ')
    else:
        return _inflect.number_to_words(num, andword='')


def _expand_ID_number(m):
    num = str(m.group(0))
    string = []
    for i in range(len(num)):
        string.append(_inflect.number_to_words(num[i]))
    return ' '.join(string)


def _expand_decimal_point(m):
    return m.group(1).replace('.', ' point ')


def normalize_numbers(text):
    """
    Normalizes numbers in an utterance as preparation for TTS
    
    text (string): Text to be preprocessed
    """
    text = re.sub(_comma_number_re, _remove_commas, text)
    text = re.sub(_pounds_re, r'\1 pounds', text)
    text = re.sub(_dollars_re, _expand_dollars, text)
    text = re.sub(_decimal_number_re, _expand_decimal_point, text)
    text = re.sub(_ordinal_re, _expand_ordinal, text)
    text = re.sub(_ID_number_re, _expand_ID_number, text)
    text = re.sub(_number_re, _expand_number, text)
    return text


# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+')

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
    ('mrs', 'misess'),
    ('mr', 'mister'),
    ('dr', 'doctor'),
    ('prof', 'professor'),
    ('univ', 'university'),
    ('st', 'saint'),
    ('co', 'company'),
    ('jr', 'junior'),
    ('maj', 'major'),
    ('gen', 'general'),
    ('drs', 'doctors'),
    ('rev', 'reverend'),
    ('lt', 'lieutenant'),
    ('hon', 'honorable'),
    ('sgt', 'sergeant'),
    ('capt', 'captain'),
    ('esq', 'esquire'),
    ('ltd', 'limited'),
    ('col', 'colonel'),
    ('ft', 'fort'),
]]


def expand_abbreviations(text):
    """
    Preprocesses a text to turn abbreviations into forms that the TTS can pronounce properly
    
    text (string): Text to be preprocessed
    """
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


# List of (word, replacement) pairs for acronym or special words:
_acronym = [
    (' a ', ' ae '),
    (' s ', ' eh s, '),
    ('array', 'ER RAY'),
    ('API', 'AE P I'),
    ('distributional', 'distributionall'),
    ('ECTS', 'E C T EH S,'),
    ('Erasmus', 'E RAS MOUS'),
    ('ID', 'I D'),
    ('IMS', 'I M EH S'),
    ('NLP', 'N L P'),
    ('PhD', 'P h D'),
    ('PWR 05B', 'Pfaffen vaald ring five B, '),
    ('psycholinguistics', 'psycho linguistics'),
    ('stuttgart', 'stu gart'),
    ('Stuttgart', 'Stu gart'),
    ('vegan', 'viygan'),
    ('Vegan', 'Viygan'),
    ('ImsLecturers', 'I M EH S Lecturers'),
    ('imsLecturers', 'I M EH S Lecturers'),
]

def expand_acronym(text):
    """
    Preprocesses a text to turn acronyms into forms that the TTS can pronounce properly
    
    text (string): Text to be preprocessed
    """
    for word, replacement in _acronym:
        text = re.sub(word, replacement, text)
    return text


def _expand_email(m):
    address = str(m.group(0))
    string = " ".join(address)
    string = string.replace('@', ', at, ').replace('.',', dot, ').replace('-', ', hyphen, ')
    return string

def expand_email(text):
    text = re.sub(_email_re, _expand_email, text)
    return text

def expand_numbers(text):
    return normalize_numbers(text)


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def transliteration_cleaners(text):
    """Pipeline for non-English text that transliterates to ASCII."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text


def english_cleaners(text):
    """Pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = collapse_whitespace(text)
    return text


def remove_unnecessary_symbols(text):
    # added
    text = re.sub(r'[()[]<>"]+', '', text)
    text = re.sub(r'/', ' ', text)
    return text


def expand_symbols(text):
    # added
    text = re.sub(";", ",", text)
    text = re.sub(":", ",", text)
    text = re.sub("-", " ", text)
    text = re.sub("&", "and", text)
    return text


def uppercase(text):
    # added
    return text.upper()


def custom_english_cleaners(text):
    """Custom pipeline for English text, including number and abbreviation expansion."""
    text = convert_to_ascii(text)
    text = expand_email(text)
    text = expand_acronym(text)
    text = lowercase(text)
    text = expand_numbers(text)
    text = expand_abbreviations(text)
    text = expand_symbols(text)
    text = remove_unnecessary_symbols(text)
    text = uppercase(text)
    text = collapse_whitespace(text)
    return text
