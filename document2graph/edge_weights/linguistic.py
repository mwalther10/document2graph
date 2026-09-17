"""Shallow linguistic resources shared by the rule-based need estimators.

Everything here works without spaCy: the lexicons and regexes below are the
default path, because German is morphologically regular enough in the places
that matter (nouns capitalize, articles are a closed class). spaCy, when it is
installed, replaces the regexes with POS tags and noun chunks and is more
accurate on definite descriptions and finite verbs; the fallback keeps the
package usable with no extra download.
"""

from __future__ import annotations

import re
from functools import lru_cache

# --- pronouns that point at something outside themselves -------------------
GERMAN_ANAPHORS = {
    "er", "sie", "es", "ihn", "ihm", "ihr", "ihnen", "sein", "seine", "seinem",
    "seinen", "seiner", "seines", "ihre", "ihrem", "ihren", "ihrer", "ihres",
    "dessen", "deren", "derer", "man",
}
ENGLISH_ANAPHORS = {
    "it", "its", "he", "him", "his", "she", "her", "hers", "they", "them",
    "their", "theirs", "one", "ones",
}

# --- demonstratives: explicit backward pointers ----------------------------
GERMAN_DEMONSTRATIVES = {
    "dieser", "diese", "dieses", "diesen", "diesem", "jener", "jene", "jenes",
    "jenen", "jenem", "solche", "solcher", "solches", "solchen", "solchem",
    "obige", "obigen", "obiger", "obige", "genannte", "genannten", "genannter",
    "letztere", "letzteren", "erstere", "ersteren", "besagte", "besagten",
    "derartige", "derartigen", "entsprechende", "entsprechenden",
}
ENGLISH_DEMONSTRATIVES = {
    "this", "these", "those", "that", "such", "said", "former", "latter",
    "aforementioned", "above",
}

# --- discourse connectives: explicit instructions to connect backward ------
GERMAN_CONNECTIVES = {
    "jedoch", "daher", "deshalb", "deswegen", "zudem", "außerdem", "ausserdem",
    "ferner", "folglich", "somit", "allerdings", "dennoch", "hingegen",
    "dagegen", "dabei", "hierbei", "hierzu", "hierfür", "dazu", "dafür",
    "darüber", "demgegenüber", "dementsprechend", "demzufolge", "insbesondere",
    "beispielsweise", "entsprechend", "andernfalls", "ansonsten", "schließlich",
    "zunächst", "anschließend", "ebenso", "ebenfalls", "trotzdem", "vielmehr",
    "zusätzlich", "weiterhin", "abschließend", "stattdessen",
}
ENGLISH_CONNECTIVES = {
    "however", "therefore", "thus", "moreover", "furthermore", "conversely",
    "hence", "consequently", "additionally", "besides", "nevertheless",
    "nonetheless", "meanwhile", "finally", "accordingly", "instead",
    "similarly", "likewise", "otherwise", "subsequently",
}
# connectives that are two words and so cannot be matched on the first token
MULTIWORD_CONNECTIVES = (
    "in contrast", "on the contrary", "on the other hand", "as a result",
    "for example", "for instance", "that said", "in addition",
    "im gegensatz", "darüber hinaus", "zum beispiel", "aus diesem grund",
    "im folgenden", "vor diesem hintergrund", "auf dieser grundlage",
)

GERMAN_ARTICLES = {"der", "die", "das", "den", "dem", "des"}
ENGLISH_ARTICLES = {"the"}

# function words used only to tell the two languages apart
_GERMAN_MARKERS = GERMAN_ARTICLES | {
    "und", "nicht", "mit", "von", "für", "auf", "ist", "sind", "ein", "eine",
    "werden", "wird", "bei", "durch", "zur", "zum", "oder", "auch", "nach",
}
_ENGLISH_MARKERS = ENGLISH_ARTICLES | {
    "and", "not", "with", "of", "for", "on", "is", "are", "a", "an",
    "be", "will", "at", "by", "to", "or", "also", "after",
}

_WORD = re.compile(r"[\wÄÖÜäöüß]+", re.UNICODE)
# German nouns capitalize, so "des Blutzuckerspiegels" is a reliable catch;
# English has no such signal and falls back to any following word
_GERMAN_DEFINITE = re.compile(
    r"\b[Dd](?:er|ie|as|en|em|es)\s+([A-ZÄÖÜ][\wÄÖÜäöüß\-]+)", re.UNICODE)
_ENGLISH_DEFINITE = re.compile(r"\b[Tt]he\s+([a-zA-Z][\w\-]+)", re.UNICODE)
# two or more capitals in a row, optionally with digits: HbA1c misses here by
# design (mixed case), so allow a trailing lowercase+digit tail
_ACRONYM = re.compile(r"\b[A-ZÄÖÜ]{2,}[a-z0-9]*[A-Z0-9]?\b|\b[A-ZÄÖÜ][a-z]?[A-Z][a-z]?\d\w*\b")
# acronyms that are units, standards or so common that no document expands them
_ACRONYM_STOPLIST = {
    "ID", "OK", "PDF", "URL", "DOI", "ISBN", "ISSN", "USA", "EU", "WHO",
    "II", "III", "IV", "VI", "VII", "VIII", "IX", "XI", "XII",
    "MG", "ML", "KG", "CM", "MM", "MMHG", "BMI",
}


def words(text: str) -> list[str]:
    return _WORD.findall(text)


def detect_language(text: str) -> str:
    """``"de"`` or ``"en"`` from function-word counts; German wins a tie.

    Deliberately crude -- it only has to pick which lexicon to use, and the two
    lexicons barely overlap.
    """
    lowered = [word.lower() for word in words(text)]
    if not lowered:
        return "de"
    german = sum(1 for word in lowered if word in _GERMAN_MARKERS)
    english = sum(1 for word in lowered if word in _ENGLISH_MARKERS)
    return "en" if english > german else "de"


def resolve_language(configured: str, text: str) -> str:
    return detect_language(text) if configured == "auto" else configured


def lexicons(language: str) -> dict[str, set[str]]:
    if language == "en":
        return {
            "anaphors": ENGLISH_ANAPHORS,
            "demonstratives": ENGLISH_DEMONSTRATIVES,
            "connectives": ENGLISH_CONNECTIVES,
            "articles": ENGLISH_ARTICLES,
        }
    return {
        "anaphors": GERMAN_ANAPHORS,
        "demonstratives": GERMAN_DEMONSTRATIVES,
        "connectives": GERMAN_CONNECTIVES,
        "articles": GERMAN_ARTICLES,
    }


def definite_descriptions(text: str, language: str) -> list[tuple[str, int]]:
    """``(head noun, character offset)`` of every definite description."""
    pattern = _ENGLISH_DEFINITE if language == "en" else _GERMAN_DEFINITE
    found = []
    for match in pattern.finditer(text):
        head = match.group(1)
        found.append((head, match.start()))
    return found


def acronyms(text: str) -> list[tuple[str, int]]:
    return [(match.group(0), match.start()) for match in _ACRONYM.finditer(text)
            if match.group(0).upper() not in _ACRONYM_STOPLIST and len(match.group(0)) >= 2]


def is_expanded_in(acronym: str, text: str) -> bool:
    """Whether ``text`` itself introduces the acronym.

    Catches both orders of the usual gloss -- ``Body-Mass-Index (BMI)`` and
    ``BMI (Body-Mass-Index)`` -- and the case where the acronym's letters are
    the initials of the words right before it.
    """
    escaped = re.escape(acronym)
    if re.search(rf"\(\s*{escaped}\s*\)", text) or re.search(rf"\b{escaped}\s*\(", text):
        return True
    initials = [char for char in acronym if char.isalpha()]
    if len(initials) < 2:
        return False
    before = text[: text.find(acronym)]
    preceding = words(before)[-len(initials):]
    if len(preceding) < len(initials):
        return False
    return all(word[:1].lower() == letter.lower()
               for word, letter in zip(preceding, initials))


def stem(word: str, length: int = 6) -> str:
    """A crude prefix stem, enough to match German inflection and compounds."""
    return word.lower()[:length]


def starts_with_connective(text: str, language: str) -> bool:
    stripped = text.lstrip().lstrip("-–—•*·").lstrip()
    lowered = stripped.lower()
    if any(lowered.startswith(phrase) for phrase in MULTIWORD_CONNECTIVES):
        return True
    first = words(stripped)[:1]
    return bool(first) and first[0].lower() in lexicons(language)["connectives"]


def starts_lowercase(text: str) -> bool:
    """Whether the first letter is lowercase.

    In German this is unusually informative: capitalization is otherwise rigid,
    and only nouns capitalize mid-sentence, so a lowercase opening means the
    unit was cut out of a running sentence.
    """
    stripped = text.lstrip().lstrip("-–—•*·§").lstrip()
    for char in stripped:
        if char.isalpha():
            return char.islower()
    return False


@lru_cache(maxsize=8)
def load_spacy(model_name: str):
    """A cached spaCy pipeline, or ``None`` when spaCy or the model is missing.

    Never raises: every caller has a regex fallback, and an optional accuracy
    upgrade should not be able to break a run.
    """
    try:
        import spacy
    except ImportError:
        return None
    try:
        return spacy.load(model_name, disable=["ner", "lemmatizer"])
    except Exception:
        return None


# Closed-class finite verb forms, the fallback for "does this unit have a finite
# verb" when spaCy is not installed. Far from complete -- it catches auxiliaries
# and modals, which carry most German clinical prose ("soll gesenkt werden",
# "ist entscheidend") but miss a lexical verb used on its own. Install the spaCy
# extra for a real answer.
GERMAN_FINITE_FALLBACK = {
    "ist", "sind", "war", "waren", "sei", "seien", "bin", "bist", "seid",
    "wird", "werden", "wurde", "wurden", "wirst", "werdet",
    "hat", "haben", "hatte", "hatten", "hast", "habt",
    "kann", "können", "konnte", "konnten", "kannst", "könnt",
    "soll", "sollen", "sollte", "sollten", "sollst", "sollt",
    "muss", "müssen", "musste", "mussten", "musst", "müsst",
    "darf", "dürfen", "durfte", "durften", "darfst", "dürft",
    "mag", "mögen", "möchte", "möchten", "will", "wollen", "wollte", "wollten",
    "gibt", "geben", "gab", "gaben", "zeigt", "zeigen", "zeigte", "zeigten",
    "besteht", "bestehen", "erfolgt", "erfolgen", "gilt", "gelten",
}
ENGLISH_FINITE_FALLBACK = {
    "is", "are", "was", "were", "am", "be", "been", "being",
    "has", "have", "had", "does", "do", "did",
    "can", "could", "shall", "should", "will", "would", "may", "might", "must",
    "shows", "show", "showed", "gives", "give", "gave", "remains", "remain",
}


def has_finite_verb(text: str, language: str, nlp=None) -> bool:
    """Whether the unit contains a finite verb.

    With spaCy this reads the morphology directly. Without it, falls back to the
    closed-class lexicons above, which under-detect lexical verbs.
    """
    if nlp is not None:
        for token in nlp(text):
            if token.pos_ in ("VERB", "AUX") and "Fin" in token.morph.get("VerbForm"):
                return True
        return False
    lexicon = ENGLISH_FINITE_FALLBACK if language == "en" else GERMAN_FINITE_FALLBACK
    return any(word.lower() in lexicon for word in words(text))


def is_fragment(text: str, language: str, nlp=None) -> bool:
    """Whether the unit fails to parse as a main clause.

    With spaCy: no token is a sentence root with a verbal head. Without it: the
    unit does not end in sentence-final punctuation, which is the surface
    symptom of the same thing for list items and table cells.
    """
    stripped = text.strip()
    if not stripped:
        return True
    if nlp is not None:
        for token in nlp(stripped):
            if token.dep_ == "ROOT" and token.pos_ in ("VERB", "AUX"):
                return False
        return True
    return stripped[-1] not in ".!?"


def ends_with_colon(text: str) -> bool:
    """Whether the text ends in a colon.

    A property of the *parent*, and the strongest single signal in this family:
    a colon is a syntactic promise that what follows completes the sentence.
    """
    return text.rstrip().endswith(":")
