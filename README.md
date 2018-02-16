# morpheval_v2

Evaluation of the morphological quality of machine translation outputs.
The automatically generated test suite in English should be translated
into French. The output is then analyzed and provides three
types of information:

* Adequacy: has the morphological information been well conveyed from the source?
* Fluency: do we have local agreement?
* Consistency: how well is the system confident in its prediction?

## Requirements

* Python 3
* Download the [test suite](https://morpheval.limsi.fr/morpheval.limsi.v2.en.sents) and [sentence tags](https://morpheval.limsi.fr/morph_test_suite_limsi.en.info).
* Download the [French dictionary](https://morpheval.limsi.fr/lefff.pkl) (taken from the [Lefff](http://alpage.inria.fr/~sagot/lefff.html))

## How To

Translate the source file `morpheval.limsi.v2.en.sents` and run the
[Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer) on it (`-no-escape` argument recommended). Then:

`python3 evaluate_fr.py -i output.tokenized -n morpheval.limsi.v2.en.info [-d /path/to/lefff.pkl]`

## Publication

Franck Burlot and François Yvon, [Evaluating the morphological competence of machine translation systems](http://www.statmt.org/wmt17/pdf/WMT05.pdf). In Proceedings of the Second Conference on Machine Translation (WMT’17). Association for Computational Linguistics, Copenhagen, Denmark, 2017.
