#!/usr/bin/python3

"""
Morpheval evaluation script for French
"""


import argparse
import pickle
import math

from collections import defaultdict, Counter
from itertools import permutations


def get_pairs(text, info):
    info = [l.split() for l in info]
    sent_id = info[0][0]  # get 1st sentence ID.
    sents = []
    sent = []
    i = 0
    for line in text:
        # tokenization fix
        new_sent = []
        for word in line.split():
            if '-' in word and word.split('-')[-1] in ['le', 'la', 'les', 'ils', 'les']:
                new_sent += word.split('-')
            else:
                new_sent.append(word)
        # lower all words (all dictionary entries are lowered)
        sents.append([w.lower() for w in new_sent])
        i += 1
        # new sentence group
        try:
            if sent_id != info[i][0]:
                yield sents, info[i-1][1]
                sents = []
                sent_id = info[i][0]
        except IndexError:
            yield sents, info[i-1][1]


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
    gender = ['m', 'f']
    number = ['s', 'p']
    person = ['1', '2', '3']
    tense = ['P', 'F', 'I', 'J', 'C', 'Y', 'S', 'T', 'K', 'G', 'W']
    # get words from the 2nd sentence that
    # are not in the 1st sentence
    index = get_new_words_idx(sents[0], sents[1])
    words = [sents[1][i] for i in index]

    if morph == 'pron2nouns':
        # both sentences are identical
        if index == []:
            return 0
        adj = [t[2] for w in words for t in d_morph[w] if t[0] == 'adj']
        noun = [t[2] for w in words for t in d_morph[w] if t[0].startswith('n')]
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
        if len(index) < 2:
            return 0
        verb1 = [t[2] for w in words for t in d_morph[w] if t[0] == 'v']
        # find rightmost second verb
        verb2 = [t[2] for i, w in enumerate(sents[1]) for t in d_morph[w] if t[0] == 'v' and i > min(index)]
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

    if morph == 'future':
        # both sentences are identical
        if index == []:
            return 0
        tags = [t[2] for w in words for t in d_morph[w] if t[0] == 'v']
        feat = [t for t in tags if 'F' in t]
        if len(tags) < 1:
            return None
        if feat != []:
            return 1
        else:
            # search for analytical future (il va venir)
            full_entries = [t for w in words for t in d_morph[w] if t[0] == 'v']
            future_analytic = False
            for entry in full_entries:
                if entry[1] == 'aller' and 'P' in entry[2]:
                    future_analytic = True
            if future_analytic:
                return 1
        return 0

    if morph in ['past', 'negation', 'noun_number', 'pron_plur', 'comparative', 'superlative', 'conditional', 'subjunctive']:
        # both sentences are identical
        if index == []:
            return 0
        if morph == 'past':
            tags = [t[2] for w in words for t in d_morph[w] if t[0] == 'v']
            feat = [t for t in tags if 'I' in t or 'J' in t or 'K' in t]
        if morph == 'conditional':
            tags = [t[2] for w in words for t in d_morph[w] if t[0] == 'v']
            feat = [t for t in tags if 'C' in t]
        if morph == 'subjunctive':
            tags = [t[2] for w in words for t in d_morph[w] if t[0] == 'v']
            if tags == []:
                # indicative and subjunctive can be similar:
                # je crois qu'il mange (indicative)
                # je ne crois pas qu'il mange (subjunctive)
                index = [i for i, _ in enumerate(sents[1])]
                words = sents[1]
                tags = [t[2] for w in words for t in d_morph[w] if t[0] == 'v']
            feat = [t for t in tags if 'S' in t or 'T' in t]
        if morph == 'noun_number':
            tags = [t[2] for w in words for t in d_morph[w] if t[0].startswith('n')]
            feat = [t for t in tags if 'p' in t]
        if morph == 'negation':
            feat = [w for w in words if w in ['ne', "n'", "non"] or w.startswith('aucun')]
        if morph == 'pron_plur':
            feat = [w for w in words if w in ['ils', 'les', 'leur', 'eux', 'en']]
        if morph == 'comparative':
            feat = [w for w in words if w in ['plus', 'moins', 'pire', 'davantage', 'mieux'] or w.startswith('ultérieur') or w.startswith('meilleur') or w.startswith('antérieur') or w.startswith('inférieur') or w.startswith('supérieur') or w.startswith('moindre')]
        if morph == 'superlative':
            feat = []
            art = []
            if ('le' in sents[1] or 'la' in sents[1] or 'les' in sents[1]) and 'plus' in words:
                art = ['ok']
            else:
                art = [w for w in words if w in ['le', 'la', 'les']]
            sup = [w for w in words if w.startswith('meilleur') or w.startswith('pire') or w == 'plus']
            if art != [] and sup != []:
                feat = ['ok']

        if morph in ['future', 'past', 'noun_number', 'conditional', 'subjunctive'] and len(tags) < 1:
            return None
        if feat != []:
            return 1
        else:
            return 0

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
        tags = [t[2] for w in words for t in d_morph[w] if t[0].startswith('n')]
        # find the leftmost new noun in the sentence, the pronoun should be after.
        index_nouns = [i for i, w in enumerate(sents[1]) for t in d_morph[w] if w in words and t[0].startswith('n')]
        if len(tags) > 0:
            tags_pron = [t[2] for i, w in enumerate(sents[1]) for t in d_morph[w] if (t[0] == 'pro' or t[0].startswith('cl')) if i > min(index_nouns)]
            if len(tags_pron) > 0:
                total += 1
                if subcat == 'gender':
                    res += _check_coref(words, tags, tags_pron, gender)
                else:
                    res += _check_coref(words, tags, tags_pron, number)

        # get variant words
        index_base = get_new_words_idx(sents[1], sents[0]) # [i for i, w in enumerate(sents[0]) if w not in sents[1]]
        if index_base == []:
            return 0, 0
        words_base = [sents[0][i] for i in index_base]
        tags = [t[2] for w in words_base for t in d_morph[w] if t[0].startswith('n')]
        # find the leftmost new noun in the sentence, the pronoun should be after.
        index_nouns = [i for i, w in enumerate(sents[0]) for t in d_morph[w] if w in words_base and t[0].startswith('n')]
        if len(tags) > 0:
            tags_pron = [t[2] for i, w in enumerate(sents[0]) for t in d_morph[w] if (t[0] == 'pro' or t[0].startswith('cl')) if i > min(index_nouns)]
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
    #print(sorted_number)
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
    gender = ['m', 'f']
    number = ['s', 'p']
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
                adj = [t[2] for t in d_morph[sents[i][idx]] if t[0] == 'adj']
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


def eval_verb(sents, morph):
    number = ['s', 'p']
    person = ['1', '2', '3']
    tense = ['P', 'F', 'I', 'J', 'C', 'Y', 'S', 'T', 'K', 'G', 'W']
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
                verb = [t[2] for t in d_morph[sents[i][idx]] if t[0] == 'v']
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
parser.add_argument('-d', dest='d', nargs="?", type=str,
                    help="french dictionary", default='lefff.pkl')
parser.add_argument('-output-fails', dest='fails', nargs="?", type=str,
                    help="output name file to store sentence pairs that have failed in any A and B sets", default=None)

args = parser.parse_args()

correct = 0
total = 0

d_morph = defaultdict(lambda: [])
for k, v in pickle.load(open(args.d, 'rb')).items():
    d_morph[k] = v

fail_file = None
if args.fails:
    fail_file = open(args.fails, 'w')

results = defaultdict(lambda: 0)
total = defaultdict(lambda: 0)

ent_adj_gend = 0
ent_adj_numb = 0
total_adj = 0
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

    elif morph == 'syns_verb':
        n, p, t = eval_verb(sents, morph)
        ent_v_nb += n
        ent_v_ps += p
        ent_v_tm += t
        total_v += 1
    
    if morph in ['pron2nouns']:
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

    else:
        res = evaluate(sents, morph, subcat)
        if res != None:
            results[morph] += res
            total[morph] += 1

    if fail_file and not morph.startswith('syns') and res == 0:
        fail_file.write('=== ' + morph + ' ===\n')
        fail_file.write(' '.join(sents[0]) + '\n')
        fail_file.write(' '.join(sents[1]) + '\n')
        
# Display results of evaluation
print("\n==== A/B-sets ====\n")
filler = 20
for res, nb in sorted(results.items()):
    res_display = res + ': '
    while len(res_display) < filler:
        res_display = ' ' + res_display
    
    print("{}{:.1f}% ({}/{})".format(res_display, nb/total[res]*100, nb, total[res]))

print("\n==== C-set ====\n")
print("== adjectives ==")
print("gender: {:.3f}".format(ent_adj_gend/total_adj))
print("number: {:.3f}".format(ent_adj_numb/total_adj))

print("== verbs ==")
print("number: {:.3f}".format(ent_v_nb/total_v))
print("person: {:.3f}".format(ent_v_ps/total_v))
print("tense/mode: {:.3f}".format(ent_v_tm/total_v))

