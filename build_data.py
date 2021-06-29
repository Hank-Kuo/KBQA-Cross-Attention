import os
import math
import argparse
from itertools import count
from rapidfuzz import fuzz, process
from collections import defaultdict

from ..utils.utils import *
from ..utils.generic_utils import normalize_answer, unique
from ..utils.freebase_utils import if_filterout
from .. import config


IGNORE_DUMMY = True
ENT_TYPE_HOP = 1
# Entity mention types: 'NP', 'ORGANIZATION', 'DATE', 'NUMBER', 'MISC', 'ORDINAL', 'DURATION', 'PERSON', 'TIME', 'LOCATION'

def build_kb_data(kb, used_fbkeys=None):
    entities = defaultdict(int)
    entity_types = defaultdict(int)
    relations = defaultdict(int)
    vocabs = defaultdict(int)
    if not used_fbkeys:
        used_fbkeys = kb.keys()
    for k in used_fbkeys:
        if not k in kb:
            continue
        v = kb[k]
        entities[v['id']] += 1
        # We prefer notable_types than type since they are more representative.
        # If notable_types are not available, we use only the first available type.
        # We found the type field contains much noise.
        selected_types = (v['notable_types'] + v['type'])[:ENT_TYPE_HOP]
        for ent_type in selected_types:
            entity_types[ent_type] += 1
        for token in [y for x in selected_types for y in x.lower().split('/')[-1].split('_')]:
            vocabs[token] += 1
        # Add entity vocabs
        selected_names = v['name'][:1] + v['alias'] # We need all topic entity alias
        for token in [y for x in selected_names for y in tokenize(x.lower())]:
            vocabs[token] += 1
        if not 'neighbors' in v:
            continue
        for kk, vv in v['neighbors'].items(): # 1st hop
            if if_filterout(kk):
                continue
            relations[kk] += 1
            # Add relation vocabs
            for token in [x for x in kk.lower().split('/')[-1].split('_')]:
                vocabs[token] += 1
            for nbr in vv:
                if isinstance(nbr, str):
                    for token in [y for y in tokenize(nbr.lower())]:
                        vocabs[token] += 1
                    continue
                elif isinstance(nbr, bool):
                    continue
                elif isinstance(nbr, float):
                    continue
                    # vocabs.update([y for y in tokenize(str(nbr).lower())])
                elif isinstance(nbr, dict):
                    nbr_k = list(nbr.keys())[0]
                    nbr_v = nbr[nbr_k]
                    entities[nbr_k] += 1
                    selected_types = (nbr_v['notable_types'] + nbr_v['type'])[:ENT_TYPE_HOP]
                    for ent_type in selected_types:
                        entity_types[ent_type] += 1
                    selected_names = (nbr_v['name'] + nbr_v['alias'])[:1]
                    for token in [y for x in selected_names for y in tokenize(x.lower())] + \
                        [y for x in selected_types for y in x.lower().split('/')[-1].split('_')]:
                        vocabs[token] += 1
                    if not 'neighbors' in nbr_v:
                        continue
                    for kkk, vvv in nbr_v['neighbors'].items(): # 2nd hop
                        if if_filterout(kkk):
                            continue
                        relations[kkk] += 1
                        # Add relation vocabs
                        for token in [x for x in kkk.lower().split('/')[-1].split('_')]:
                            vocabs[token] += 1
                        for nbr_nbr in vvv:
                            if isinstance(nbr_nbr, str):
                                for token in [y for y in tokenize(nbr_nbr.lower())]:
                                    vocabs[token] += 1
                                continue
                            elif isinstance(nbr_nbr, bool):
                                continue
                            elif isinstance(nbr_nbr, float):
                                # vocabs.update([y for y in tokenize(str(nbr_nbr).lower())])
                                continue
                            elif isinstance(nbr_nbr, dict):
                                nbr_nbr_k = list(nbr_nbr.keys())[0]
                                nbr_nbr_v = nbr_nbr[nbr_nbr_k]
                                entities[nbr_nbr_k] += 1
                                selected_types = (nbr_nbr_v['notable_types'] + nbr_nbr_v['type'])[:ENT_TYPE_HOP]
                                for ent_type in selected_types:
                                    entity_types[ent_type] += 1
                                selected_names = (nbr_nbr_v['name'] + nbr_nbr_v['alias'])[:1]
                                for token in [y for x in selected_names for y in tokenize(x.lower())] + \
                                    [y for x in selected_types for y in x.lower().split('/')[-1].split('_')]:
                                    vocabs[token] += 1
                            else:
                                raise RuntimeError('Unknown type: %s' % type(nbr_nbr))
                else:
                    raise RuntimeError('Unknown type: %s' % type(nbr))
    return (entities, entity_types, relations, vocabs)

def build_qa_vocab(qa):
    vocabs = defaultdict(int)
    for each in qa:
        for token in tokenize(each['qText'].lower()):
            vocabs[token] += 1
    return vocabs

def delex_query_topic_ent(query, topic_ent, ent_types):
    query = tokenize(query.lower())
    if topic_ent == '':
        return query, None

    ent_type_dict = {}
    for ent, type_ in ent_types:
        if ent not in ent_type_dict:
            ent_type_dict[ent] = type_
        else:
            if ent_type_dict[ent] == 'NP':
                ent_type_dict[ent] = type_

    ret = process.extract(topic_ent.replace('_', ' '), set(list(zip(*ent_types))[0]), scorer=fuzz.token_sort_ratio)

    if len(ret) == 0:
        return query, None

    # We prefer Non-NP entity mentions
    # e.g., we prefer `uk` than `people in the uk` when matching `united_kingdom`
    topic_men = None
    topic_score = None
    for token, score in ret:
        if ent_type_dict[token].lower() in config.topic_mention_types:
            topic_men = token
            topic_score = score
            break

    if topic_men is None:
        return query, None

    topic_ent_type = ent_type_dict[topic_men].lower()
    topic_tokens = tokenize(topic_men.lower())
    indices = [i for i, x in enumerate(query) if x == topic_tokens[0]]
    for i in indices:
        if query[i: i + len(topic_tokens)] == topic_tokens:
            start_idx = i
            end_idx = i + len(topic_tokens)
            break
    query_template = query[:start_idx] + [topic_ent_type] + query[end_idx:]
    #print(query_template, topic_men)
    return query_template, topic_men

def delex_query(query, ent_mens, mention_types):
    for men, type_ in ent_mens:
        type_ = type_.lower()
        if type_ in mention_types:
            men = tokenize(men.lower())
            indices = [i for i, x in enumerate(query) if x == men[0]]
            start_idx = None
            for i in indices:
                if query[i: i + len(men)] == men:
                    start_idx = i
                    end_idx = i + len(men)
                    break
            if start_idx is not None:
                query = query[:start_idx] + ['__{}__'.format(type_)] + query[end_idx:]
    return query

def build_data(qa, kb, entity2id, entityType2id, relation2id, vocab2id, pred_seed_ents=None):
    queries = []
    raw_queries = []
    query_mentions = []
    memories = []
    cand_labels = [] # Candidate answer labels (i.e., names)
    gold_ans_labels = [] # True gold answer labels
    gold_ans_inds = [] # The "gold" answer indices corresponding to the cand list
    for qid, each in enumerate(qa):
        freebase_key = each['freebaseKey'] if not pred_seed_ents else pred_seed_ents[qid]
        if isinstance(freebase_key, list):
            freebase_key = freebase_key[0] if len(freebase_key) > 0 else ''
        # Convert query to query template
        query, topic_men = delex_query_topic_ent(each['qText'], freebase_key, each['entities'])
        query2 = delex_query(query, each['entities'], config.delex_mention_types)
        q = [vocab2id[x] if x in vocab2id else config.RESERVED_TOKENS['UNK'] for x in query2]
        queries.append(q)
        raw_queries.append(query)

        query_mentions.append([(tokenize(x[0].lower()), x[1].lower()) for x in each['entities'] if topic_men != x[0]])
        gold_ans_labels.append(each['answers'])

        if not freebase_key in kb:
            gold_ans_inds.append([])
            memories.append([[]] * 8)
            cand_labels.append([])
            continue

        ans_cands = build_ans_cands(kb[freebase_key], entity2id, entityType2id, relation2id, vocab2id)
        memories.append(ans_cands[:-1])
        cand_labels.append(ans_cands[-1])
        if len(ans_cands[0]) == 0:
            gold_ans_inds.append([])
            continue

        norm_cand_labels = [normalize_answer(x) for x in ans_cands[-1]]
        tmp_cand_inds = []
        for a in each['answers']:
            a = normalize_answer(a)
            # Find all the candidiate answers which match the gold answer.
            inds = [i for i, j in zip(count(), norm_cand_labels) if j == a]
            tmp_cand_inds.extend(inds)
        # Note that tmp_cand_inds can be empty in which case
        # the question can *NOT* be answered by this KB entity.
        gold_ans_inds.append(tmp_cand_inds)
    return (queries, raw_queries, query_mentions, memories, cand_labels, gold_ans_inds, gold_ans_labels)
