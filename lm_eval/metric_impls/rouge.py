"""
MIT License

Copyright (c) 2021 GEM-metrics authors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

MODIFICATION: This implementation is based on the GEM-metrics implementation.
    https://github.com/GEM-benchmark/GEM-metrics/blob/431a8174bd6b3637e8d6118bfad2983e39e99733/gem_metrics/rouge.py
"""
import functools
import logging
import nltk
import numpy as np
import re
import typing
from rouge_score import rouge_scorer


logger = logging.getLogger(__name__)


class NltkWordTokenizer:

    def __init__(self, lang: str = None, download_dir: str = None):
        """
        :param lang: pycountry.db.Language
            pycountry.db.Language object representing the language (result of pycountry.languages.get)
            If `lang` is not specified, the default naive tokenizer will be used.
        """
        self.tokenizer = self._nltk_tokenizer(lang)
        self.download_dir = download_dir

    def tokenize(self, text: str) -> typing.List[str]:
        return self.tokenizer(text)

    def _nltk_tokenizer(self, lang: str) -> typing.Callable:
        """ Based on: https://github.com/GEM-benchmark/GEM-metrics/blob/431a8174bd6b3637e8d6118bfad2983e39e99733/gem_metrics/tokenize.py#L10
        
        Return the default tokenizer function for a given language (Punkt, backoff to dumb_tokenize).
        The functions takes one argument (text) and reutrns a list of tokens.
        
        :param lang: pycountry.db.Language
            pycountry.db.Language object representing the language (result of pycountry.languages.get)
        """
        self._nltk_ensure_download("tokenizers/punkt")
        tokenizer = NltkWordTokenizer._default_tokenizer
        if lang is not None:
            try:
                tokenizer = functools.partial(
                    nltk.tokenize.word_tokenize,
                    language=lang.name.lower())
                # NOTE: This will trigger an exception if Punkt doesn't have the language
                tokenizer(".")
            except LookupError:
                logger.warning(f"NLTK Punkt does not support language `{lang.name}`; using the default naive tokenizer.")
                tokenizer = NltkWordTokenizer._default_tokenizer  # punkt
        return tokenizer

    def _nltk_ensure_download(self, package: str):
        import nltk
        """Check if the given package is available, download if needed."""
        try:
            nltk.data.find(package)
        except LookupError:
            package_id = re.sub("^[^/]*/", "", package)
            nltk.download(package_id, download_dir=self.download_dir)

    @staticmethod
    def _default_tokenizer(text: str) -> typing.List[str]:
        """ Based on: https://github.com/GEM-benchmark/GEM-metrics/blob/431a8174bd6b3637e8d6118bfad2983e39e99733/gem_metrics/tokenize.py#L28
        A naive tokenizer that separates tokens by spaces as a language agnostic default. 

        :param text: String to be tokenized
        """
        import re

        toks = text
        # separate quotes everywhere
        toks = re.sub(
            r'(["<>{}“”«»–|—„‚‘]|\[|\]|``|\'\'|‘‘|\^)', r" \1 ", toks)

        # the following characters (double-characters) are separated everywhere (except inside URLs)
        toks = re.sub(r"([;!()?#\$£%&*…]|--)", r" \1 ", toks)

        # short hyphen is separated if it is followed or preceeded by non-alphanuneric character and
        # is not a part of --, or a unary minus
        toks = re.sub(r"([^\-\w])\-([^\-0-9])", r"\1 - \2", toks)
        toks = re.sub(
            r"([0-9]\s+)\-([0-9])", r"\1 - \2", toks
        )  # preceded by a number - not a unary minus
        toks = re.sub(r"([^\-])\-([^\-\w])", r"\1 - \2", toks)

        # plus is separated everywhere, except at the end of a word (separated by a space) and as unary plus
        toks = re.sub(r"(\w)\+(\w)", r"\1 + \2", toks)
        toks = re.sub(r"([0-9]\s*)\+([0-9])", r"\1 + \2", toks)
        toks = re.sub(r"\+([^\w\+])", r"+ \1", toks)

        # apostrophe is separated if it is followed or preceeded by non-alphanumeric character,
        # is not part of '', and is not followed by a digit (e.g. '60).
        toks = re.sub(r"([^\'’\w])([\'’])([^\'’\d])", r"\1 \2 \3", toks)
        toks = re.sub(r"([^\'’])([\'’])([^\'’\w])", r"\1 \2 \3", toks)

        # dot, comma, slash, and colon are separated if they do not connect two numbers
        toks = re.sub(r"(\D|^)([\.,:\/])", r"\1 \2", toks)
        toks = re.sub(r"([\.,:\/])(\D|$)", r"\1 \2", toks)

        # three dots belong together
        toks = re.sub(r"\.\s*\.\s*\.", r"...", toks)

        # most common contractions
        # I'm, I've etc.
        toks = re.sub(r"([\'’´])(s|m|d|ll|re|ve)\s", r" \1\2 ", toks)
        toks = re.sub(r"(n[\'’´]t\s)", r" \1 ", toks)  # do n't

        # other contractions, as implemented in Treex
        toks = re.sub(r" ([Cc])annot\s", r" \1an not ", toks)
        toks = re.sub(r" ([Dd])\'ye\s", r" \1\' ye ", toks)
        toks = re.sub(r" ([Gg])imme\s", r" \1im me ", toks)
        toks = re.sub(r" ([Gg])onna\s", r" \1on na ", toks)
        toks = re.sub(r" ([Gg])otta\s", r" \1ot ta ", toks)
        toks = re.sub(r" ([Ll])emme\s", r" \1em me ", toks)
        toks = re.sub(r" ([Mm])ore\'n\s", r" \1ore \'n ", toks)
        toks = re.sub(r" \'([Tt])is\s", r" \'\1 is ", toks)
        toks = re.sub(r" \'([Tt])was\s", r" \'\1 was ", toks)
        toks = re.sub(r" ([Ww])anna\s", r" \1an na ", toks)

        # clean extra space
        toks = re.sub(r"\s+", " ", toks)
        toks = toks.strip()
        return toks.split(" ")


DEFAULT_NLTK_TOKENIZER = NltkWordTokenizer()


def rouge(
    refs: typing.List[str],
    pred: str,
    rouge_types: typing.List[str], 
    tokenizer = DEFAULT_NLTK_TOKENIZER,
):
    """ ROUGE with multi-reference support

    Implementation based on GEM-metrics:
    https://github.com/GEM-benchmark/GEM-metrics/blob/431a8174bd6b3637e8d6118bfad2983e39e99733/gem_metrics/rouge.py

    TODO: Add newline split support. `rouge-score==0.0.4` expects `pred` and
    `refs` sentences to be split with newlines in order to compute `rougeLsum` scores.

    :param refs:
        A `list` of reference `str`s.
    :param pred:
        A single prediction `str`s.
    :param rouge_types:
        A `list` of ROUGE types to score, from the set:
        {"rouge1", "rouge2", "rougeL", "rougeLsum"}
    :param tokenize:
        Any tokenizer object with a `tokenize` method.
    """
    pred = " ".join(tokenizer.tokenize(pred))
    refs = [" ".join(tokenizer.tokenize(ref)) for ref in refs]

    scorer = rouge_scorer.RougeScorer(
        rouge_types=rouge_types,
        use_stemmer=True
    )

    # ROUGE multi-ref jackknifing
    if len(refs) > 1:
        cur_scores = [scorer.score(ref, pred) for ref in refs]

        # get best score for all leave-one-out sets
        best_scores = []
        for leave in range(len(refs)):
            cur_scores_leave_one = [
                cur_scores[s] for s in range(len(refs)) if s != leave
            ]
            best_scores.append(
                {
                    rouge_type: max(
                        [s[rouge_type] for s in cur_scores_leave_one],
                        key=lambda s: s.fmeasure,
                    )
                    for rouge_type in rouge_types
                }
            )
        # average the leave-one-out bests to produce the final score
        score = {
            rouge_type: rouge_scorer.scoring.Score(
                np.mean([b[rouge_type].precision for b in best_scores]),
                np.mean([b[rouge_type].recall for b in best_scores]),
                np.mean([b[rouge_type].fmeasure for b in best_scores]),
            )
            for rouge_type in rouge_types
        }
    else:
        score = scorer.score(refs[0], pred)
    # convert the named tuples to plain nested dicts
    score = {
        rouge_type: {
            "precision": score[rouge_type].precision,
            "recall": score[rouge_type].recall,
            "fmeasure": score[rouge_type].fmeasure,
        }
        for rouge_type in rouge_types
    }
    return score
