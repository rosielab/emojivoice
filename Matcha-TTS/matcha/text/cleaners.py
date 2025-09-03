""" from https://github.com/keithito/tacotron

Cleaners are transformations that run over the input text at both training and eval time.

Cleaners can be selected by passing a comma-delimited list of cleaner names as the "cleaners"
hyperparameter. Some cleaners are English-specific. You'll typically want to use:
  1. "english_cleaners" for English text
  2. "transliteration_cleaners" for non-English text that can be transliterated to ASCII using
     the Unidecode library (https://pypi.python.org/pypi/Unidecode)
  3. "basic_cleaners" if you do not want to transliterate (in this case, you should also update
     the symbols in symbols.py to match your data).
"""

import logging
import re

import phonemizer
from unidecode import unidecode
from misaki import ja
import os

# Set logging to critical to avoid excessive output
critical_logger = logging.getLogger("phonemizer")
critical_logger.setLevel(logging.CRITICAL)

# Force Espeak-NG to use a fixed temporary directory
ESPEAK_TMPDIR = "/tmp/espeak_ng_tmp"
os.makedirs(ESPEAK_TMPDIR, exist_ok=True)
os.environ["TMPDIR"] = ESPEAK_TMPDIR  # Set TMPDIR globally

# Initialize phonemizers only once at module load
global_phonemizers = {
    "en-us": phonemizer.backend.EspeakBackend(
        language="en-us",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
        logger=critical_logger,
    ),
    "fr-fr": phonemizer.backend.EspeakBackend(
        language="fr-fr",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
        logger=critical_logger,
    ),
    "es": phonemizer.backend.EspeakBackend(
        language="es",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
        logger=critical_logger,
    ),
    "de": phonemizer.backend.EspeakBackend(
        language="de",
        preserve_punctuation=True,
        with_stress=True,
        language_switch="remove-flags",
        logger=critical_logger,
    ),
}

# Initialize Japanese phonemizer once
global_japanese_phonemizer = ja.JAG2P()

def get_phonemizer(language):
    """Retrieve the pre-initialized phonemizer for the specified language."""
    return global_phonemizers.get(language, None)

def get_japanese_phonemizer():
    """Retrieve the pre-initialized Japanese phonemizer."""
    return global_japanese_phonemizer

# Regular expression matching whitespace:
_whitespace_re = re.compile(r"\s+")

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations_en = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("mrs", "misess"),
        ("ms", "miss"),
        ("mr", "mister"),
        ("dr", "doctor"),
        ("st", "saint"),
        ("co", "company"),
        ("jr", "junior"),
        ("maj", "major"),
        ("gen", "general"),
        ("drs", "doctors"),
        ("rev", "reverend"),
        ("lt", "lieutenant"),
        ("hon", "honorable"),
        ("sgt", "sergeant"),
        ("capt", "captain"),
        ("esq", "esquire"),
        ("ltd", "limited"),
        ("col", "colonel"),
        ("ft", "fort"),
    ]
]

_abbreviations_fr = [
    (re.compile("\\b%s\\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("m.", "monsieur"),
        ("dr", "docteur"),
        ("st", "saint"),
    ]
]

# List of (regular expression, replacement) pairs for German abbreviations
_abbreviations_de = [
    (re.compile(r"\b%s\." % x[0], re.IGNORECASE), x[1])
    for x in [
        ("hr", "herr"),
        ("fr", "frau"),
        ("dr", "doktor"),
        ("prof", "professor"),
        ("bsp", "beispiel"),
        ("usw", "und so weiter"),
        ("z", "zu"),
        ("z.b", "zum beispiel"),
        ("ca", "zirka"),
        ("bzw", "beziehungsweise"),
        ("d.h", "das heißt"),
        ("u.a", "unter anderem"),
        ("u.u", "unter umständen"),
        ("u.v.m", "und vieles mehr"),
        ("vgl", "vergleiche"),
    ]
]

# List of character replacements with specific conditions
_replacements_ja = [
    (re.compile(r"(?<!\s)\.(?!\s)"), " てん"),  # Replace "." if NOT followed by space
    (re.compile(r"-(?=\d)"), " えん"),  # Replace "-" only if followed by a number
    (re.compile(r"%"), " パーセント"),  # Always replace "%"
    (re.compile(r"@"), " アットマーク"),  # Always replace "@"
    (re.compile(r"\\\\"), " バックスラッシュ"),  # Always replace "\\" (needs escaping)
    (re.compile(r"/"), " スラッシュ"),  # Always replace "/"
    (re.compile(r"\$"), " ドル"),  # Always replace "$"
    (re.compile(r"€"), " ユーロ"),  # Always replace "€"
    (re.compile(r"¥"), " えん"),  # Always replace "¥"
    (re.compile(r"\+"), " プラス"),  # Always replace "+"
    (re.compile(r"="), " イコール")  # Always replace "="
]


_replacements_en = [
    (re.compile(r"\.\.\."), "ELLIPSIS_MARKER"),  # Temporarily replace ellipsis
    (re.compile(r"\$(\d+)\.(\d+)"), r"\1 dollars and \2 cents"),  # "$5.45" → "5 dollars 45 cents"
    (re.compile(r"€(\d+)\.(\d+)"), r"\1 euros and \2 cents"),  # "€5.45" → "5 euros 45 cents"
    (re.compile(r"¥(\d+)\.(\d+)"), r"\1 yen and \2 cents"),  # "¥5.45" → "5 yen 45 cents"

    (re.compile(r"(?<=\D)\.(?=\D)(?!\s)", re.IGNORECASE), " dot "),  # "." between letters → "dot"
    (re.compile(r"(?<=\d)\.(?=\d)(?!\s)"), " point "),  # "." between numbers → "point"

    (re.compile(r"\$(\d+)"), r"\1 dollars"),  # "$5" → "5 dollars"
    (re.compile(r"€(\d+)"), r"\1 euros"),  # "€5" → "5 euros"
    (re.compile(r"¥(\d+)"), r"\1 yen"),  # "¥5" → "5 yen"
    (re.compile(r"ELLIPSIS_MARKER"), "..."),  # Restore ellipsis
]

#add mius back slashes, slash, equals
# List of character replacements with specific conditions, phd to doctorant
_replacements_fr = [
    (re.compile(r"\.\.\."), "ELLIPSIS_MARKER"),  # Temporarily replace ellipsis
    (re.compile(r"\("), ""),
    (re.compile(r"\)"), ""),
    (re.compile(r"(\d+)\.(\d+)\$"), r"\1 dollars et \2 centimes"),
    (re.compile(r"(\d+)\.(\d+)€"), r"\1 euros et \2 centimes"),
    (re.compile(r"(\d+)\.(\d+)¥"), r"\1 yen et \2 centimes"),
    (re.compile(r"(?<=\D)\.(?=\D)(?!\s)", re.IGNORECASE), " point "),
    (re.compile(r"(?<=\d)\,(?=\d)(?!\s)"), " vergule "),
    (re.compile(r"€"), " euros"),
    (re.compile(r"¥"), " yen"),
    (re.compile(r"Mme"), "madame"),
    (re.compile(r"Mlle"), "mademoiselle"),
    (re.compile(r"="), " égales "),
    (re.compile(r"/"), " slash "),
    (re.compile(r"-(?=\d)(?!\s)"), "négatif "),
    (re.compile(r"ELLIPSIS_MARKER"), "..."),  # Restore ellipsis
]

_replacements_de = [
    (re.compile(r"\.\.\."), "ELLIPSIS_MARKER"),  # Temporarily replace ellipsis
    (re.compile(r"\("), ""),
    (re.compile(r"\)"), ""),
    (re.compile(r"(\d+)\.(\d+)\$"), r"\1 Dollar und \2 Cent"),
    (re.compile(r"(\d+)\.(\d+)€"), r"\1 Euro und \2 Cent"),
    (re.compile(r"(\d+)\.(\d+)¥"), r"\1 Yen und \2 Sen"),  # "Sen" is the subdivision of Yen
    (re.compile(r"(?<=\D)\.(?=\D)(?!\s)", re.IGNORECASE), " Punkt "),
    (re.compile(r"(?<=\d)\,(?=\d)(?!\s)"), " Komma "),
    (re.compile(r"€"), " Euro"),
    (re.compile(r"¥"), " Yen"),
    (re.compile(r"Mme"), "Frau"),
    (re.compile(r"Mlle"), "Fräulein"),
    (re.compile(r"="), " gleich "),
    (re.compile(r"/"), " Schrägstrich "),
    (re.compile(r"-(?=\d)(?!\s)"), "minus "),
    (re.compile(r"ELLIPSIS_MARKER"), "..."),  # Restore ellipsis
]

def apply_replacements(text, language):
    """Applies the character replacements"""
    if language == "en":
        replacements = _replacements_en
    elif language == "ja":
        replacements = _replacements_ja
    elif language == "fr":
        replacements = _replacements_fr
    elif language == "de":
        replacements = _replacements_de
    for regex, replacement in replacements:
        text = regex.sub(replacement, text)
    return text

def expand_abbreviations(text, language):
    if language == "en":
        abbv= _abbreviations_en
    elif language == "fr":
        abbv = _abbreviations_fr
    elif language == "de":
        abbv = _abbreviations_de
    for regex, replacement in abbv:
        text = re.sub(regex, replacement, text)
    return text

def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text)


def convert_to_ascii(text):
    return unidecode(text)


def basic_cleaners(text):
    """Basic pipeline that lowercases and collapses whitespace without transliteration."""
    text = lowercase(text)
    text = collapse_whitespace(text)
    return text

def english_cleaners2(text):
    """Pipeline for English text, including abbreviation expansion. + punctuation + stress"""
    global_phonemizer = get_phonemizer("en-us")
    text = text.encode("utf-8").decode("utf-8")
    text = lowercase(text)
    text = expand_abbreviations(text, "en")
    text = apply_replacements(text, "en")
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes

def french_cleaners(text):
    """Pipeline for French text"""
    global_phonemizer = get_phonemizer("fr-fr")
    text = text.encode("utf-8").decode("utf-8")
    text = lowercase(text)
    text = expand_abbreviations(text, "fr")
    text = apply_replacements(text, "fr")
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes

def german_cleaners(text):
    """Pipeline for French text"""
    global_phonemizer = get_phonemizer("de")
    text = text.encode("utf-8").decode("utf-8")
    text = lowercase(text)
    text = expand_abbreviations(text, "de")
    text = apply_replacements(text, "de")
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes

def japanese_cleaners(text):
    """Pipeline for Japansese text must convert from kanji to full hiragana for the phonemizer"""
    global_japanese_phonemizer = get_japanese_phonemizer()
    text = text.encode("utf-8").decode("utf-8")
    text = apply_replacements(text, "ja")
    phonemes = global_japanese_phonemizer(text)
    phonemes = collapse_whitespace(phonemes[0])

    return phonemes

def spanish_cleaners(text):
    """Pipeline for Spanish text"""
    global_phonemizer = get_phonemizer("es")
    text = text.encode("utf-8").decode("utf-8")
    text = lowercase(text)
    text = expand_abbreviations(text, "es")
    text = apply_replacements(text, "es")
    phonemes = global_phonemizer.phonemize([text], strip=True, njobs=1)[0]
    phonemes = collapse_whitespace(phonemes)
    return phonemes