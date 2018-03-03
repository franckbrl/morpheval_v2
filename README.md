# morpheval_v2

Evaluation of the morphological quality of machine translation outputs.
The automatically generated test suite in English should be translated
into one of the supported target languages (French, Czech). The output
is then analyzed and provides three types of information:

* Adequacy: has the morphological information been well conveyed from the source?
* Fluency: do we have local agreement?
* Consistency: how well is the system confident in its prediction?

## Requirements

* Python3
* Download the [test suite](https://morpheval.limsi.fr/morpheval.limsi.v2.en.sents) and [sentence tags](https://morpheval.limsi.fr/morpheval.limsi.v2.en.info)
* (French) Download the [dictionary](https://morpheval.limsi.fr/lefff.pkl) (taken from the [Lefff](http://alpage.inria.fr/~sagot/lefff.html))
* (Czech) Download and install [Morphodita](https://github.com/ufal/morphodita/releases/tag/v1.3.0) version 1.3, as well as the [dictionary](https://lindat.mff.cuni.cz/repository/xmlui/handle/11858/00-097C-0000-0023-68D8-1)

## How To

Translate the source file `morpheval.limsi.v2.en.sents` and run the
[Moses tokenizer](https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer) on it (with arguments `-no-escape` and `-l {fr|cs}`). Then:

### French

`python3 evaluate_fr.py -i output.tokenized -n morpheval.limsi.v2.en.info [-d /path/to/lefff.pkl]`

### Czech

`cat output.tokenized | tr ' ' '\n' | morphodita/src/run_morpho_analyze dictionary --input=vertical --output=vertical > output.analysis` <br>
`python3 evaluate_cs.py -i output.analysis -n morpheval.limsi.v2.en.info`

## Publication

Franck Burlot and François Yvon, [Evaluating the morphological competence of machine translation systems](http://www.statmt.org/wmt17/pdf/WMT05.pdf). In Proceedings of the Second Conference on Machine Translation (WMT’17). Association for Computational Linguistics, Copenhagen, Denmark, 2017.
