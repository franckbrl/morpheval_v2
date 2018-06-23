#!/usr/bin/python3

import argparse
import math

from collections import Counter
from collections import defaultdict
from itertools import permutations


def get_pairs(text, info):
    info = [l.split() for l in info]
    sent_id = info[0][0]  # get 1st sentence ID.
    sents = []
    tags = []
    sent = []
    tag = []
    i = 0
    for line in text:
        if line == '\n':
            sents.append(sent)
            tags.append(tag)
            sent = []
            tag = []
            # new sentence
            i += 1
            continue
        line = line.split()
        # new sentence group
        if sent_id != info[i][0]:
            yield sents, tags, info[i-1][1], info[i-1][0]
            sents = []
            tags = []
            sent_id = info[i][0]
        # add words and tags
        sent.append(line[0])
        ii = 2
        t = []
        while ii < len(line):
            t.append(line[ii])
            ii += 2
        tag.append(t)
    # last sentence
    yield sents, tags, info[-1][1], info[i-1][0]


def get_new_words_idx(base, variant):
    """
    Which words in variant are not found in base?
    """
    new_words = []
    for i, word in enumerate(variant):
        if word not in base:
            new_words.append(i)
        # a word has been duplicated in the variant
        elif base.count(word) < variant.count(word):
            new_words.append(i)
    return list(set(new_words))


def get_entropy(index, feature_list):
    # compute frequency of feature values
    counter = Counter()
    for idx in index:
        for val in index[idx]:
            if val == 'X':
                for v in feature_list:
                    counter[v] += 1
            else:
                counter[val] += 1


    # sort the values by frequency (then by key)
    sorted_number = sorted(counter.items(), key=lambda x: (x[1],x[0]), reverse=True)

    final_values = []
    for idx in index:
        if len(index[idx]) == 0:
            # No new word with the expected PoS was found in the sentence.
            final_values.append('u')
            continue
        for nb, _ in sorted_number:
            if nb in index[idx] or 'X' in index[idx]:
                final_values.append(nb)
                break

    # Spread 'u' count into different predictions
    # (getting 5 times 'u' should not be a good thing)
    val_qnt = defaultdict(lambda: 0)
    i = 0
    for val in final_values:
        if val == 'u':
            val_qnt['u'+str(i)] = 1
            i += 1
        else:
            val_qnt[val] += 1

    # compute normalized entropy
    max_ent = - math.log(5)
    ent = sum([((val_qnt[n]/5) * math.log(val_qnt[n]/5)) for n in val_qnt])

    return ent/max_ent


def eval_noun(sents, tags, morph):
    case = ['1', '2', '3', '4', '5', '6', '7']
    # Get new word index for each sentence
    # that is compared to all the others.
    index_case = {}
    for i, _ in enumerate(sents):
        index_case[i] = set()
        for j, _ in enumerate(sents):
            if j == i:
                continue
            for idx in get_new_words_idx(sents[j], sents[i]):
                noun = [t for t in tags[i][idx] if t[0] == 'N']
                # find case
                for feat in noun:
                    feat = feat[4]
                    index_case[i].add(feat)

    # case entropies
    ent_c = get_entropy(index_case, case)

    return ent_c


def eval_adj(sents, tags, morph):
    gender = ['F', 'H', 'M', 'N', 'Q', 'T', 'Z']
    number = ['P', 'S', 'W']
    case = ['1', '2', '3', '4', '5', '6', '7']
    # find words to evaluate
    index_gender = {}
    index_number = {}
    index_case = {}
    for i, _ in enumerate(sents):
        index_gender[i] = set()
        index_number[i] = set()
        index_case[i] = set()
        for j, _ in enumerate(sents):
            if j == i:
                continue
            for idx in get_new_words_idx(sents[j], sents[i]):
                adj = [t for t in tags[i][idx] if t[0] == 'A']
                # find gender
                for feat in adj:
                    feat = feat[2]
                    if feat in ['I', 'M', 'Y']:
                        feat = 'M'
                    index_gender[i].add(feat)
                # find number
                for feat in adj:
                    feat = feat[3]
                    if feat in ['P', 'D']:
                        feat = 'P'
                    index_number[i].add(feat)
                # find case
                for feat in adj:
                    feat = feat[4]
                    index_case[i].add(feat)

    # entropies
    ent_g = get_entropy(index_gender, gender)
    ent_n = get_entropy(index_number, number)
    ent_c = get_entropy(index_case, case)

    return ent_g, ent_n, ent_c
                    
def eval_verb(sents, tags, morph):
    number = ['P', 'S', 'W']
    person = ['1', '2', '3']
    tense = ['F', 'H', 'P', 'R']
    negation = ['A', 'N']
    # find words to evaluate
    index_number = {}
    index_person = {}
    index_tense = {}
    index_negation = {}
    for i, sent in enumerate(sents):
        index_number[i] = set()
        index_person[i] = set()
        index_tense[i] = set()
        index_negation[i] = set()
        for j, sent_comp in enumerate(sents):
            if j == i:
                continue
            for idx in get_new_words_idx(sents[j], sents[i]):
                verb = [t for t in tags[i][idx] if t[0] == 'V']
                # find number
                for feat in verb:
                    feat = feat[3]
                    if feat in ['P', 'D']:
                        feat = 'P'
                    index_number[i].add(feat)
                # person
                for feat in verb:
                    feat = feat[7]
                    index_person[i].add(feat)
                # tense
                for feat in verb:
                    feat = feat[8]
                    index_tense[i].add(feat)
                # negation
                for feat in verb:
                    feat = feat[10]
                    index_negation[i].add(feat)

    # entropies
    ent_nb = get_entropy(index_number, number)
    ent_ps = get_entropy(index_person, person)                    
    ent_tm = get_entropy(index_tense, tense)                    
    ent_ne = get_entropy(index_negation, negation)                    

    return ent_nb, ent_ps, ent_tm, ent_ne


def evaluate(sents, tags, morph, subcat=None):
    # get words from the 2nd sentence that
    # are not in the 1st sentence
    index = get_new_words_idx(sents[0], sents[1])
    # both sentences are identical
    if index == []:
        return 0

    if morph == 'pron_relative':
        def _check_coref(tags, tags_pron):
            inter = list(set(tags).intersection(tags_pron))
            if inter != [] or 'X' in tags or 'X' in tags_pron:
                return 1
            return 0

        def _search_compare(sent, tags, index, subcat):
            res = 0
            gender = ['F', 'H', 'M', 'N', 'Q', 'T', 'Z']
            number = ['P', 'S', 'W']
            tags_nouns = [t for i in index for t in tags[i] if t.startswith('N')]
            # find the leftmost new noun in the sentence, the pronoun should be after.
            index_nouns = [i for i, _ in enumerate(sent) for t in tags[i] if i in index and t.startswith('N')]
            if len(tags_nouns) > 0:
                tags_pron = [t for i, _ in enumerate(sent) for t in tags[i] if (t[0] == 'P' and t[1] in ['4', '9', 'E', 'J', 'K', 'Q', 'Y']) if i > min(index_nouns)]
                if len(tags_pron) > 0:
                    if subcat == 'gender':
                        tags_nouns = [t[2] for t in tags_nouns]
                        tags_pron = [t[2] for t in tags_pron]
                        # merge similar fine-grained tags
                        tags_nouns = ['M' if f in ['I', 'Y'] else f for f in tags_nouns]
                        tags_pron = ['M' if f in ['I', 'Y'] else f for f in tags_pron]
                        res += _check_coref(tags_nouns, tags_pron)
                    elif subcat == 'number':
                        tags_nouns = [t[3] for t in tags_nouns]
                        tags_pron = [t[3] for t in tags_pron]
                        # merge similar fine-grained tags
                        tags_nouns = ['P' if f == 'D' else f for f in tags_nouns]
                        tags_pron = ['P' if f == 'D' else f for f in tags_pron]
                        res += _check_coref(tags_nouns, tags_pron)
            return res
            
        # compute indices of words from base that are not in variant
        index_base = get_new_words_idx(sents[1], sents[0])
        res = 0
        # process both base and variant sentences
        res += _search_compare(sents[1], tags[1], index, subcat)
        res += _search_compare(sents[0], tags[0], index_base, subcat)

        return res

    if morph == 'coref':
        def _check_coref(tags, tags_pron):
            inter = list(set(tags).intersection(tags_pron))
            if inter != [] or 'X' in tags or 'X' in tags_pron:
                return 1
            return 0

        def _search_compare(sent, tags, index, subcat):
            res = 0
            gender = ['F', 'H', 'M', 'N', 'Q', 'T', 'Z']
            number = ['P', 'S', 'W']
            tags_nouns = [t for i in index for t in tags[i] if t.startswith('N')]
            # find the leftmost new noun in the sentence, the pronoun should be after.
            index_nouns = [i for i, _ in enumerate(sent) for t in tags[i] if i in index and t.startswith('N')]
            if len(tags_nouns) > 0:
                tags_pron = [t for i, _ in enumerate(sent) for t in tags[i] if (t[0] == 'P' and t[0] in ['5', 'D', 'H', 'P']) if i > min(index_nouns)]
                if len(tags_pron) > 0:
                    if subcat == 'gender':
                        tags_nouns = [t[2] for t in tags_nouns]
                        tags_pron = [t[2] for t in tags_pron]
                        # merge similar fine-grained tags
                        tags_nouns = ['M' if f in ['I', 'Y'] else f for f in tags_nouns]
                        tags_pron = ['M' if f in ['I', 'Y'] else f for f in tags_pron]
                        res += _check_coref(tags_nouns, tags_pron)
            return res
            
        # compute indices of words from base that are not in variant
        index_base = get_new_words_idx(sents[1], sents[0])
        res = 0
        # process both base and variant sentences
        res += _search_compare(sents[1], tags[1], index, subcat)
        res += _search_compare(sents[0], tags[0], index_base, subcat)

        return res

    if morph == 'preposition':
        # words in sentence 2 that are not in 1 and vice-versa.
        index1 = get_new_words_idx(sents[0], sents[1])
        index2 = get_new_words_idx(sents[1], sents[0])
        res1 = 0
        for i in index1:
            case_prep = [t[4] for t in tags[1][i] if t[0] == 'R']
            if case_prep == []:
                continue
            # look for the noun on the left
            if i < len(sents[1])-1:
                for j, tag in enumerate(tags[1][i+1:], i+1):
                    tag = [t for t in tag if t[0] == 'N']
                    if tag != []:
                        case_noun = [t[4] for t in tag]
                        if list(set(case_prep).intersection(case_noun)) != [] or 'X' in case_noun:
                            res1 = 1

        res2 = 0
        for i in index2:
            case_prep = [t[4] for t in tags[0][i] if t[0] == 'R']
            if case_prep == []:
                continue
            # look for the noun on the left
            if i < len(sents[0])-1:
                for j, tag in enumerate(tags[0][i+1:], i+1):
                    tag = [t for t in tag if t[0] == 'N']
                    if tag != []:
                        case_noun = [t[4] for t in tag]
                        if list(set(case_prep).intersection(case_noun)) != [] or 'X' in case_noun:
                            res2 = 1

        return res1 + res2

    if morph == 'coordverb':
        new_verb = []
        for i in index:
            tag_is = list(set([(i, (t[7], t[3], t[8], t[1])) for t in tags[1][i] if t[0] == 'V' and t[-1] == '-']))
            new_verb.append(tag_is)
        new_verb = [n for n in new_verb if n != []]
        if new_verb == []:
            return 0
        n = ['person', 'number', 'tense'].index(subcat)
        for word in new_verb:
            for analysis in word:
                i = analysis[0]
                tag = analysis[1][n]
                # go to the right and find the second verb
                for j, tag2 in enumerate(tags[1][i+1:], i+1):
                    tag_right = list(set([(t[7], t[3], t[8], t[1]) for t in tag2 if t[0] == 'V' and t[-1] == '-']))
                    if tag_right == []:
                        continue
                    for tag_r_full in tag_right:
                        tag_r = tag_r_full[n]
                        # Present perfective forms
                        if tag_r == tag or tag_r == 'X' or tag == 'X':
                            if subcat == 'tense' and tag_r == 'P' and (tag_r_full[-1] == 'B' or analysis[1][-1] == 'B'):
                                if tag_r_full[-1] == analysis[1][-1] == 'B':
                                    return 1
                                else:
                                    continue
                            if subcat == 'tense' and (tag_r_full[-1] == 'f' or analysis[1][-1] == 'f'):
                                if tag_r_full[-1] == analysis[1][-1]:
                                    return 1
                                else:
                                    continue
                            return 1
        return 0

    if morph == 'pron2coord':
        noun = []
        for i in index:
            tag_is = list(set([t[4] for t in tags[1][i] if t[0] == 'N' and t[-1] == '-']))
            noun.append(tag_is)
        noun = [n for n in noun if n != []]
        if len(noun) < 2:
            return 0
        done = []
        for n1, n2 in permutations(noun, 2):
            if (n2, n1) in done:
                continue
            done.append((n1, n2))
            if 'X' in n1 + n2:
                return 1
            inter = list(set(n1).intersection(n2))
            if inter != []:
                continue
            else:
                return 1
        return 0

    if morph == 'pron2nouns':
        adj = [t[2:5] for i, w in enumerate(tags[1]) for t in w if t[0] == 'A' and i in index and t[-1] == '-']
        noun = [t[2:5] for i, w in enumerate(tags[1]) for t in w if t[0] == 'N' and i in index and t[-1] == '-']
        if adj == [] or noun == []:
            return 0
        n = ['gender', 'number', 'case'].index(subcat)
        feat_adj = [t[n] for t in adj]
        feat_noun = [t[n] for t in noun]

        inter = list(set(feat_adj).intersection(feat_noun))
        if inter != []:
            return 1
        if 'X' in feat_adj + feat_noun:
            return 1
        else:
            return 0

    for i in index:

        if morph == 'future':
            feature = [t[8] for t in tags[1][i] if t[0] == 'V']
            if 'F' in feature:
                return 1
            else:
                # present form of perfective verb
                pos = [t[0] for t in tags[1][i]]
                subpos = [t[1] for t in tags[1][i]]
                feature = [t[8] for t in tags[1][i]]
                for p, s, f in zip(pos, subpos, feature):
                    if p == 'V' and s == 'B' and f == 'P':
                        return 1

        elif morph == 'past':
            feature = [t[8] for t in tags[1][i] if t[0] == 'V']
            if 'R' in feature or 'H' in feature or 'X' in feature:
                return 1

        elif morph == 'conditional':
            feature = [t[1] for t in tags[1][i] if t[0] == 'V']
            if 'c' in feature:
                return 1

        elif morph == 'comparative':
            feature = [t[9] for t in tags[1][i] if t[0] == 'A']
            if '2' in feature or 'X' in feature:
                return 1

        elif morph == 'superlative':
            feature = [t[9] for t in tags[1][i] if t[0] == 'A']
            if '3' in feature or 'X' in feature:
                return 1

        elif morph == 'negation':
            feature = [t[10] for t in tags[1][i] if t[0] == 'V']
            if 'N' in feature:
                return 1   

        elif morph == 'noun_number':
            feature = [t[3] for t in tags[1][i] if t[0] == 'N']
            if 'P' in feature or 'D' in feature or 'X' in feature:
                return 1  

        elif morph == 'pron_fem':
            feature = [t[2] for t in tags[1][i] if t[0] == 'P']
            if 'F' in feature or 'Q' in feature or 'H' in feature or 'T' in feature or 'X' in feature:
                return 1    

        elif morph == 'pron_plur':
            feature = [t[3] for t in tags[1][i] if t[0] == 'P']
            if 'P' in feature or 'D' in feature or 'W' in feature or 'X' in feature:
                return 1   

    return 0


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='i', nargs="?", type=argparse.FileType('r'),
                    help="input morphodita analysis")
parser.add_argument('-n', dest='n', nargs="?", type=argparse.FileType('r'),
                    help="input info file")
parser.add_argument('-l', '--latex', dest='latex', action='store_true',
                    help="output in latex format")
args = parser.parse_args()

correct = 0
total = 0

ent_n = 0
total_n = 0
ent_adj_gend = 0
ent_adj_numb = 0
ent_adj_case = 0
total_adj = 0
ent_v_nb = 0
ent_v_ps = 0
ent_v_tm = 0
ent_v_ne = 0
total_v = 0

results = defaultdict(lambda: 0)
total = defaultdict(lambda: 0)

for sents, tags, morph, sent_id in get_pairs(args.i, args.n):
    if morph == 'future':
        # Remove bad sentence pairs evaluating future and
        # containing "until" (in morpheval_v2 test suite).
        if sent_id in ['313']:
            print("sentences removed:", morph, sent_id)
            continue

    if len(sents) == len(tags) == 2 and not morph.startswith('syns'):
        subcat = None
        if ':' in morph:
            morph, subcat = morph.split(':')
            if subcat == 'time':
                subcat = 'tense'

        if morph in ['pron2nouns']:
            for subcat in ['gender', 'number', 'case']:
                inf = morph+'-'+subcat
                results[inf] += evaluate(sents, tags, morph, subcat)
                total[inf] += 1

        elif morph in ['coordverb']:
            for subcat in ['person', 'number', 'tense']:
                inf = morph+'-'+subcat
                results[inf] += evaluate(sents, tags, morph, subcat)
                total[inf] += 1

        elif morph in ['preposition']:
            results[morph] += evaluate(sents, tags, morph, subcat)
            total[morph] += 2

        elif morph == 'coref':
            for subcat in ['gender']:
                inf = morph+'-'+subcat
                results[inf] += evaluate(sents, tags, morph, subcat)
                total[inf] += 2

        elif morph == 'pron_relative':
            for subcat in ['gender', 'number']:
                inf = morph+'-'+subcat
                results[inf] += evaluate(sents, tags, morph, subcat)
                total[inf] += 2
                
        else:
            results[morph] += evaluate(sents, tags, morph, subcat)
            total[morph] += 1

    else:
        if morph == 'syns_noun':
            ent_n += eval_noun(sents, tags, morph)
            total_n += 1

        elif morph == 'syns_adj':
            g, n, c = eval_adj(sents, tags, morph)
            ent_adj_gend += g
            ent_adj_numb += n
            ent_adj_case += c
            total_adj += 1

        elif morph == 'syns_verb':
            nb, ps, tm, ne = eval_verb(sents, tags, morph)
            ent_v_nb += nb
            ent_v_ps += ps
            ent_v_tm += tm
            ent_v_ne += ne
            total_v += 1

# Display results of evaluation
if args.latex:
    a_feat = ['past',
    'future',
    'conditional',
    'negation',
    'pron_fem',
    'pron_plur',
    'noun_number',
    'comparative',
    'superlative'
    ]
    meanA = sum([results[m]/total[m]*100 for m in a_feat])/len(a_feat)
    latexA = [results[m]/total[m]*100 for m in a_feat] + [meanA]
    print("A-set: {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% \\\\ ".format(*latexA))

    b_feat = ['coordverb-number',
    'coordverb-person',
    'coordverb-tense',
    'pron2coord',
    'pron2nouns-gender',
    'pron2nouns-number',
    'pron2nouns-case',
    'pron_relative-gender',
    'pron_relative-number',
    'preposition',
    'coref-gender'
    ]
    meanB = sum([results[m]/total[m]*100 for m in b_feat])/len(b_feat)
    latexB = [results[m]/total[m]*100 for m in b_feat] + [meanB]
    print("B-set: {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% & {:.1f}\% \\\\ ".format(*latexB))

    mean=sum([ent_n/total_n, ent_adj_gend/total_adj, ent_adj_numb/total_adj, ent_adj_case/total_adj, ent_v_nb/total_v, ent_v_ps/total_v, ent_v_tm/total_v, ent_v_ne/total_v]) / 8.
    latex=[ent_n/total_n, ent_adj_gend/total_adj, ent_adj_numb/total_adj, ent_adj_case/total_adj, ent_v_nb/total_v, ent_v_ps/total_v, ent_v_tm/total_v, ent_v_ne/total_v, mean]

    print("C-set: {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} & {:.3f} \\\\ ".format(*latex))

else:
    print("==== A/B-sets ====\n")
    filler = 22
    for res, nb in sorted(results.items()):
        if res in ['adj_strong', 'compounds_syns', 'subjunctive', 'verb_position']:
            continue
        res_display = res + ': '
        while len(res_display) < filler:
            res_display = ' ' + res_display
        print("{}{:.1f}% ({}/{})".format(res_display, nb/total[res]*100, nb, total[res]))

    print("\n==== C-set ====\n")
    print("* nouns")
    print("    case: {:.3f}".format(ent_n/total_n))
    print("\n* adjectives")
    print("  gender: {:.3f}".format(ent_adj_gend/total_adj))
    print("  number: {:.3f}".format(ent_adj_numb/total_adj))
    print("    case: {:.3f}".format(ent_adj_case/total_adj))
    print("\n* verbs")
    print("  number: {:.3f}".format(ent_v_nb/total_v))
    print("  person: {:.3f}".format(ent_v_ps/total_v))
    print("   tense: {:.3f}".format(ent_v_tm/total_v))
    print("negation: {:.3f}".format(ent_v_ne/total_v))
