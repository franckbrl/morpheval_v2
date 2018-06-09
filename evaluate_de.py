#!/usr/bin/python3

"""
Morpheval evaluation script for German
"""


import argparse
import pickle
import math
import re

from collections import defaultdict, Counter
from itertools import permutations


def get_pairs(text, info):
    info = [l.split() for l in info]
    sent_id = info[0][0]  # get 1st sentence ID.
    sents = []
    sent = []
    i = 0
    for line in text:
        # lower all words (all dictionary entries are lowered)
        sents.append([w.lower() for w in line.split()])
        i += 1
        # new sentence group
        try:
            if sent_id != info[i][0]:
                yield sents, info[i-1][1]
                sents = []
                sent_id = info[i][0]
        except IndexError:
            yield sents, info[i-1][1]


def read_smor(smored):
    d_morph = defaultdict(lambda: [])
    d_compounds = defaultdict(lambda: [])
    morph = []
    compounds = []
    word = None
    for line in smored:
        if line.startswith('analyze>'):
            if word:
                d_compounds[word] = compounds
                d_morph[word] = morph
                compounds = []
                morph = []
            try:
                word = line.split()[1].lower()
            except IndexError:
                word = None
        else:
            if line.startswith('no result for'):
                continue
            if word:
                # get tags
                pattern = re.compile("<([^>]*)>")
                match = pattern.findall(line)
                # Main PoS start with '+' (only one per ta sequence)
                idx = [i for i, t in enumerate(match) if t.startswith('+')][0]
                tags = match[idx:]
                morph.append(tags)
                # get compound split
                line = line.rstrip()
                for tag in match:
                    line = line.replace('<' + tag + '>', ' ')
                comp = line.split()
                if comp not in compounds:
                    compounds.append(comp)

    d_compounds[word] = compounds
    d_morph[word] = morph

    return d_morph, d_compounds


def find_features(pos, morph):
    """
    Return the value of the morphological feature.
    If the feature has no value, return 'x' (any value).
    """
    feats = set()
    for p in pos:
        for feat in morph:
            if feat in p:
                feats.add(feat)
    if len(feats) == 0:
        feats.add('x')
    return feats


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


def evaluate(sents, morph, subcat=None):
    gender = ['Masc', 'Fem', 'Neut']
    number = ['Sg', 'Pl']
    person = ['1', '2', '3']
    tense = ['Pres', 'Past', 'Imp']
    # get words from the 2nd sentence that
    # are not in the 1st sentence
    index = get_new_words_idx(sents[0], sents[1])
    words = [sents[1][i] for i in index]

    if morph == 'pron2nouns':
        # both sentences are identical
        if index == []:
            return 0
        adj = [t for w in words for t in d_morph[w] if t[0] == '+ADJ']
        noun = [t for w in words for t in d_morph[w] if len(t) > 1 and t[1] == 'Subst']
        # no dictionary entry
        if adj == [] or noun == []:
            return None
        if subcat == 'gender':
            feat_adj = find_features(adj, gender)
            feat_noun = find_features(noun, gender)
        elif subcat == 'number':
            feat_adj = find_features(adj, number)
            feat_noun = find_features(noun, number)
        inter = list(feat_adj.intersection(feat_noun))
        if inter != [] or 'x' in feat_adj or 'x' in feat_noun:
            return 1
        else:
            return 0
    
    if morph == 'coordverb':
        # both sentences are identical
        if index == []:
            return 0
        verb1 = [t for w in words for t in d_morph[w] if t[0] == '+V']
        # find rightmost second verb
        verb2 = [t for i, w in enumerate(sents[1]) for t in d_morph[w] if t[0] == '+V' and i > min(index)]
        # no dictionary entry
        if verb1 == [] or verb2 == []:
            return None
        if subcat == 'person':
            feat1 = find_features(verb1, person)
            feat2 = find_features(verb2, person)
        elif subcat == 'number':
            feat1 = find_features(verb1, number)
            feat2 = find_features(verb2, number)
        elif subcat == 'tense':
            feat1 = find_features(verb1, tense)
            feat2 = find_features(verb2, tense) 
        inter = list(feat1.intersection(feat2))
        if inter != [] or 'x' in feat1 or 'x' in feat2:
            return 1
        else:
            return 0


    if morph == 'compounds_syns':
        # both sentences are identical
        if index == []:
            return 0
        # compare base to variant also
        index_base = get_new_words_idx(sents[1], sents[0])
        words_base = [sents[0][i] for i in index_base]

        compounds_base = [c for w in words_base for c in d_compounds[w] if len(c) > 1]
        compounds_variant = [c for w in words for c in d_compounds[w] if len(c) > 1 and c not in compounds_base]

        # compounds are not in dictionary. Compute simple character similarity.
        # TODO: get more compound examples from big data.
        if compounds_base == [] or compounds_variant == []:
            return None

        else:
            for comp_b in compounds_base:
                for comp_v in compounds_variant:
                    if len(set(comp_b).intersection(set(comp_v))) > 0:
                        return 1
        return 0

    if morph == 'verb_position':
        # Look for verbs in both sentences
        verbs_base_idx = set([i for i, w in enumerate(sents[0]) for t in d_morph[w] if t[0] == '+V'])
        verbs_variant_idx = set([i for i, w in enumerate(sents[1]) for t in d_morph[w] if t[0] == '+V'])
        # verbs not found
        if len(verbs_base_idx) == 0 or len(verbs_variant_idx) == 0:
            return None
        # base verb must be closer to the end of the
        # sentence than variant verb
        for vb in verbs_base_idx:
            for vv in verbs_variant_idx:
                base_dist = len([w for w in sents[0][vb:] if w.isalpha()])
                variant_dist = len([w for w in sents[1][vv:] if w.isalpha()])
                if base_dist < variant_dist:
                    return 1

        return 0


    if morph in ['future', 'past', 'negation', 'noun_number', 'pron_plur', 'comparative', 'superlative', 'conditional', 'adj_strong']:
        # both sentences are identical
        if index == []:
            return 0
        if morph == 'future':
            tags = [t for w in words for t in d_morph[w] if t[0] == '+V']
            feat = [w.lower() for w in words if w.startswith('werd') or w == 'wird']
        if morph == 'past':
            tags = [t[1:] for w in words for t in d_morph[w] if t[0] == '+V']
            feat = [subt for t in tags for subt in t if 'Past' in subt]
        if morph == 'conditional':
            tags = [t[1:] for w in words for t in d_morph[w] if t[0] == '+V']
            feat = [t for t in tags if 'Past' in t and 'Subj' in t]
        if morph == 'noun_number':
            tags = [t for w in words for t in d_morph[w] if len(t) > 1 and t[1] == 'Subst']
            feat = [t for t in tags if 'Pl' in t]
        if morph == 'negation':
            feat = [w for w in words if w in ['nicht'] or w.startswith('kein')]
        if morph == 'pron_plur':
            tags = [t[1:] for w in words for t in d_morph[w] if t[0] == '+PPRO']
            feat = [t for t in tags if 'Pl' in t]
        if morph == 'comparative':
            tags = [t for w in words for t in d_morph[w] if t[0] == '+ADJ']
            feat = [t for t in tags if 'Comp' in t]
        if morph == 'superlative':
            tags = [t for w in words for t in d_morph[w] if t[0] == '+ADJ']
            feat = [t for t in tags if 'Sup']
        if morph == 'adj_strong':
            tags = [t for w in words for t in d_morph[w] if t[0] == '+ADJ']
            feat = [t for t in tags if 'St']

        if morph in ['future', 'past', 'noun_number', 'conditional', 'pron_plur', 'comparative', 'superlative', 'adj_strong'] and len(tags) < 1:
            return None
        if feat != []:
            return 1
        else:
            return 0

    if morph == 'pron_relative':
        def _check_coref(tags, tags_pron):
            inter = list(set(tags).intersection(tags_pron))
            if inter != [] or 'x' in tags or 'x' in tags_pron:
                return 1
            return 0

        def _search_compare(sent, index, subcat):
            res = 0
            gender = ['Masc', 'Fem', 'Neut']
            number = ['Sg', 'Pl']
            tags_nouns = [t for w in sent for t in d_morph[w] if len(t) > 1 and t[1] == 'Subst']
            # find the leftmost new noun in the sentence, the pronoun should be after.
            index_nouns = [i for i, w in enumerate(sent) for t in d_morph[w] if w in sent and len(t) > 1 and t[1] == 'Subst']
            if len(tags_nouns) > 0:
                tags_pron = [t for i, w in enumerate(sent) for t in d_morph[w] if t[0] == '+REL' and i > min(index_nouns)]
                if len(tags_pron) > 0:
                    if subcat == 'gender':
                        tags_nouns = [t for ts in tags_nouns for t in ts if t in ['Masc', 'Fem', 'Neut', 'NoGend']]
                        tags_pron = [t for ts in tags_pron for t in ts if t in ['Masc', 'Fem', 'Neut', 'NoGend']]
                        res += _check_coref(tags_nouns, tags_pron)
                    elif subcat == 'number':
                        tags_nouns = [t for ts in tags_nouns for t in ts if t in number]
                        tags_pron = [t for ts in tags_pron for t in ts if t in number]
                        res += _check_coref(tags_nouns, tags_pron)
            return res

        # compute indices of words from base that are not in variant
        index_base = get_new_words_idx(sents[1], sents[0])
        res = 0
        # process both base and variant sentences
        res += _search_compare(sents[1], index, subcat)
        res += _search_compare(sents[0], index_base, subcat)

        return res

    if morph == 'coref':
        def _check_coref(words, tags, tags_pron, attribute):
            # find noun
            feat = find_features(tags, attribute)
            # find pronoun
            feat_pron = find_features(tags_pron, attribute)
            inter = list(feat.intersection(feat_pron))
            if inter != [] or 'x' in feat or 'x' in feat_pron:
                return 1
            return 0

        # both sentences are identical
        if index == []:
            return 0, 0
        res = 0
        total = 0

        # process both base and variant sentences
        tags = [t for w in words for t in d_morph[w] if len(t) > 1 and t[1] == 'Subst']
        # find the leftmost new noun in the sentence, the pronoun should be after.
        index_nouns = [i for i, w in enumerate(sents[1]) for t in d_morph[w] if w in words and len(t) > 1 and t[1] == 'Subst']
        if len(tags) > 0:
            tags_pron = [t for i, w in enumerate(sents[1]) for t in d_morph[w] if t[0] == '+PPRO' and i > min(index_nouns)]
            if len(tags_pron) > 0:
                total += 1
                if subcat == 'gender':
                    res += _check_coref(words, tags, tags_pron, gender)
                else:
                    res += _check_coref(words, tags, tags_pron, number)

        # get variant words
        index_base = get_new_words_idx(sents[1], sents[0])
        if index_base == []:
            return 0, 0
        words_base = [sents[0][i] for i in index_base]
        tags = [t for w in words_base for t in d_morph[w] if len(t) > 1 and t[1] == 'Subst']
        # find the leftmost new noun in the sentence, the pronoun should be after.
        index_nouns = [i for i, w in enumerate(sents[0]) for t in d_morph[w] if w in words_base and len(t) > 1 and t[1] == 'Subst']
        if len(tags) > 0:
            tags_pron = [t[2] for i, w in enumerate(sents[0]) for t in d_morph[w] if t[0] == '+PPRO' and i > min(index_nouns)]
            if len(tags_pron) > 0:
                total += 1
                if subcat == 'gender':
                    res += _check_coref(words_base, tags, tags_pron, gender)
                else:
                    res += _check_coref(words_base, tags, tags_pron, number)

        return res, total

    return None


def get_entropy(index, feature_list):
    # compute frequency of feature values
    counter = Counter()
    for idx in index:
        for val in index[idx]:
            if val == 'x':
                for v in feature_list:
                    counter[v] += 1
            else:
                counter[val] += 1
    # sort the values by frequency
    sorted_number = sorted(counter.items(), key=lambda x: (x[1],x[0]), reverse=True)
    final_values = []
    for idx in index:
        if len(index[idx]) == 0:
            # No new word with the expected PoS was found in the sentence.
            final_values.append('u')
            continue
        for nb, _ in sorted_number:
            if nb in index[idx] or 'x' in index[idx]:
                final_values.append(nb)
                break

    # Spread 'u' count into different predictions
    # (predicting 5 times 'u' should not be a good thing)
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


def eval_adj(sents, morph):
    gender = ['Masc', 'Fem', 'Neut']
    number = ['Sg', 'Pl']
    # Get new word index for each sentence
    # that is compared to all the others.
    index_gender = {}
    index_number = {}
    for i, sent in enumerate(sents):
        index_gender[i] = set() 
        index_number[i] = set() 
        for j, sent_comp in enumerate(sents):
            if j == i:
                continue
            for idx in get_new_words_idx(sents[j], sents[i]):
                w = sents[i][idx]
                adj = [t for t in d_morph[w] if t[0] == '+ADJ']
                # find gender
                for feat in find_features(adj, gender):
                    index_gender[i].add(feat)
                    # if we have 'x' and an actual tag, keep
                    # only the actual tag.
                    if 'x' in index_gender[i] and len(index_gender[i]) > 1:
                        index_gender[i].remove('x')
                # find number
                for feat in find_features(adj, number):
                    index_number[i].add(feat)
                    if 'x' in index_number[i] and len(index_number[i]) > 1:
                        index_number[i].remove('x')

    # gender and number entropies
    ent_g = get_entropy(index_gender, gender)
    ent_n = get_entropy(index_number, number)

    return ent_g, ent_n


def eval_noun(sents, morph):
    case = ['Nom', 'Acc', 'Dat', 'Gen']
    # Get new word index for each sentence
    # that is compared to all the others.
    index_case = {}
    for i, sent in enumerate(sents):
        index_case[i] = set() 
        for j, sent_comp in enumerate(sents):
            if j == i:
                continue
            for idx in get_new_words_idx(sents[j], sents[i]):
                w = sents[i][idx]
                noun = [t for t in d_morph[w] if len(t) > 1 and t[1] == 'Subst']
                # find case
                for feat in find_features(noun, case):
                    index_case[i].add(feat)
                    # if we have 'x' and an actual tag, keep
                    # only the actual tag.
                    if 'x' in index_case[i] and len(index_case[i]) > 1:
                        index_case[i].remove('x')

    # case and number entropies
    ent_c = get_entropy(index_case, case)

    return ent_c


def eval_verb(sents, morph):
    number = ['Sg', 'Pl']
    person = ['1', '2', '3']
    tense = ['Pres', 'Past', 'Imp']
    # Get new word index for each sentence
    # that is compared to all the others.
    index_number = {}
    index_person = {}
    index_tense = {}
    for i, sent in enumerate(sents):
        index_number[i] = set() 
        index_person[i] = set() 
        index_tense[i] = set() 
        for j, sent_comp in enumerate(sents):
            if j == i:
                continue

            for idx in get_new_words_idx(sents[j], sents[i]):
                w = sents[i][idx]
                verb = [t for t in d_morph[w] if t[0] == '+V']
                # find number
                for feat in find_features(verb, number):
                    index_number[i].add(feat)
                    if 'x' in index_number[i] and len(index_number[i]) > 1:
                        index_number[i].remove('x')
                # find person
                for feat in find_features(verb, person):
                    index_person[i].add(feat)
                    if 'x' in index_person[i] and len(index_person[i]) > 1:
                        index_person[i].remove('x')
                # find tense
                for feat in find_features(verb, tense):
                    index_tense[i].add(feat)
                    if 'x' in index_tense[i] and len(index_tense[i]) > 1:
                        index_tense[i].remove('x')

    # compute entropies
    ent_n = get_entropy(index_number, number)
    ent_p = get_entropy(index_person, person)
    ent_t = get_entropy(index_tense, tense)

    return ent_n, ent_p, ent_t


parser = argparse.ArgumentParser()
parser.add_argument('-i', dest='i', nargs="?", type=argparse.FileType('r'),
                    help="input sentences")
parser.add_argument('-n', dest='n', nargs="?", type=argparse.FileType('r'),
                    help="input info file")
parser.add_argument('-d', dest='d', nargs="?", type=argparse.FileType('r'),
                    help="Smored vocabulary")
args = parser.parse_args()

correct = 0
total = 0

d_morph, d_compounds = read_smor(args.d)

results = defaultdict(lambda: 0)
total = defaultdict(lambda: 0)

ent_adj_gend = 0
ent_adj_numb = 0
total_adj = 0
ent_n = 0
total_n = 0
ent_v_nb = 0
ent_v_ps = 0
ent_v_tm = 0
total_v = 0

for sents, morph in get_pairs(args.i, args.n):

    subcat = None
    if ':' in morph:
        morph, subcat = morph.split(':')
    if subcat == 'time':
        subcat = 'tense'

    if morph == 'syns_adj':
        g, n = eval_adj(sents, morph)
        ent_adj_gend += g
        ent_adj_numb += n
        total_adj += 1

    elif morph == 'syns_noun':
        ent_n += eval_noun(sents, morph)
        total_n += 1

    elif morph == 'syns_verb':
        n, p, t = eval_verb(sents, morph)
        ent_v_nb += n
        ent_v_ps += p
        ent_v_tm += t
        total_v += 1
    
    elif morph in ['pron2nouns']:
        for subcat in ['gender', 'number']:
            inf = morph+'-'+subcat
            res = evaluate(sents, morph, subcat)
            if res != None:
                results[inf] += res
                total[inf] += 1

    elif morph in ['coordverb']:
        for subcat in ['person', 'number', 'tense']:
            inf = morph+'-'+subcat
            res = evaluate(sents, morph, subcat)
            if res != None:
                results[inf] += res
                total[inf] += 1

    elif morph == 'coref':
        for subcat in ['gender']:
            inf = morph+'-'+subcat
            res, tot = evaluate(sents, morph, subcat)
            results[inf] += res
            total[inf] += tot

    elif morph == 'pron_relative':
        for subcat in ['gender', 'number']:
            inf = morph+'-'+subcat
            results[inf] += evaluate(sents, morph, subcat)
            total[inf] += 2

    else:
        res = evaluate(sents, morph, subcat)
        if res != None:
            results[morph] += res
            total[morph] += 1

# Display results of evaluation
print("\n==== A/B-sets ====\n")
filler = 22
for res, nb in sorted(results.items()):
    if res in ['subjunctive']:
        continue
    res_display = res + ': '
    while len(res_display) < filler:
        res_display = ' ' + res_display
    print("{}{:.1f}% ({}/{})".format(res_display, nb/total[res]*100, nb, total[res]))

print("\n==== C-set ====\n")
print("== adjectives ==")
print("    gender: {:.3f}".format(ent_adj_gend/total_adj))
print("    number: {:.3f}".format(ent_adj_numb/total_adj))

print("== nouns ==")
print("      case: {:.3f}".format(ent_n/total_n))

print("== verbs ==")
print("    number: {:.3f}".format(ent_v_nb/total_v))
print("    person: {:.3f}".format(ent_v_ps/total_v))
print("tense/mode: {:.3f}".format(ent_v_tm/total_v))

