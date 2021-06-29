import os
import math
import argparse
from collections import defaultdict
from itertools import count
from rapidfuzz import fuzz, process

import utils

ENT_TYPE_HOP = 1
IGNORE_DUMMY = True

RESERVED_TOKENS = {'NE': 0, 'PAD': 1, 'UNK': 2}
RESERVED_ENTS = {'PAD': 0, 'UNK': 1}
RESERVED_ENT_TYPES = {'PAD': 0, 'UNK': 1}
RESERVED_RELS = {'PAD': 0, 'UNK': 1}

extra_vocab_tokens = ['alias', 'true', 'false', 'num', 'bool', 'NE']
extra_ent_types = ['num', 'bool']

parser = argparse.ArgumentParser(description='KBQA-Cross Attention Preprocess')
parser.add_argument("--dataset_dir", default="./data", help="Path to dataset.")
parser.add_argument("--topn", default=15, help="top n candidates.")
parser.add_argument("--min_freq", default=1, help="min word vocab freq.")



def if_filterout(s):
    if s.endswith('has_sentences') or \
        s.endswith('exceptions') or s.endswith('sww_base/source') or \
        s.endswith('kwtopic/assessment'):
        return True
    else:
        return False

def build_kb_data(kb, used_fbkeys=None):
    entities = defaultdict(int)
    entity_types = defaultdict(int)
    relations = defaultdict(int)

    if not used_fbkeys:
        used_fbkeys = kb.keys()
    
    # 從 candidation entity to build dict
    for k in used_fbkeys:
        if not k in kb:
            continue
        v = kb[k]
        entities[v['id']] += 1

        selected_types = (v['notable_types'] + v['type'])[:ENT_TYPE_HOP]
        for ent_type in selected_types:
            entity_types[ent_type] += 1
        
        if not 'neighbors' in v:
            continue

        # span node to each neighbors and collect 2 hop entity and relation  
        for kk, vv in v['neighbors'].items(): # 1st hop
            if if_filterout(kk):
                continue
            relations[kk] += 1
            for nbr in vv:
                if isinstance(nbr, str):
                    continue
                elif isinstance(nbr, bool):
                    continue
                elif isinstance(nbr, float):
                    continue
                if isinstance(nbr, dict):
                    nbr_k = list(nbr.keys())[0]
                    nbr_v = nbr[nbr_k]
                    
                    entities[nbr_k] += 1
                    selected_types = (nbr_v['notable_types'] + nbr_v['type'])[:ENT_TYPE_HOP]
                    for ent_type in selected_types:
                        entity_types[ent_type] += 1

                    if not 'neighbors' in nbr_v:
                        continue
                    for kkk, vvv in nbr_v['neighbors'].items(): # 2nd hop
                        if if_filterout(kkk):
                            continue
                        relations[kkk] += 1
                        # Add relation vocabs
                        for nbr_nbr in vvv:
                            if isinstance(nbr_nbr, str):
                                continue
                            elif isinstance(nbr_nbr, bool):
                                continue
                            elif isinstance(nbr_nbr, float):
                                continue
                            elif isinstance(nbr_nbr, dict):
                                nbr_nbr_k = list(nbr_nbr.keys())[0]
                                nbr_nbr_v = nbr_nbr[nbr_nbr_k]
                                entities[nbr_nbr_k] += 1
                                selected_types = (nbr_nbr_v['notable_types'] + nbr_nbr_v['type'])[:ENT_TYPE_HOP]
                                for ent_type in selected_types:
                                    entity_types[ent_type] += 1
                            else:
                                raise RuntimeError('Unknown type: %s' % type(nbr_nbr))
                else:
                    raise RuntimeError('Unknown type: %s' % type(nbr))
    return (entities, entity_types, relations)

def build_qa_vocab(qa):
    vocabs = defaultdict(int)
    for each in qa:
        for token in utils.tokenize(each['qText'].lower()):
            vocabs[token] += 1
    return vocabs

def build_vocab(data, freebase, used_fbkeys=None, min_freq=1):
    entities, entity_types, relations = build_kb_data(freebase, used_fbkeys)

    # Entity
    all_entities = set({ent for ent in entities if entities[ent] >= min_freq})
    entity2id = dict(zip(all_entities, range(len(RESERVED_ENTS), len(all_entities) + len(RESERVED_ENTS))))
    for ent, idx in RESERVED_ENTS.items():
        entity2id.update({ent: idx})

    # Entity type
    all_ent_types = set({ent_type for ent_type in entity_types if entity_types[ent_type] >= min_freq})
    all_ent_types.update(extra_ent_types)
    entityType2id = dict(zip(all_ent_types, range(len(RESERVED_ENT_TYPES), len(all_ent_types) + len(RESERVED_ENT_TYPES))))
    for ent_type, idx in RESERVED_ENT_TYPES.items():
        entityType2id.update({ent_type: idx})

    # Relation
    all_relations = set({rel for rel in relations if relations[rel] >= min_freq})
    relation2id = dict(zip(all_relations, range(len(RESERVED_RELS), len(all_relations) + len(RESERVED_RELS))))
    for rel, idx in RESERVED_RELS.items():
        relation2id.update({rel: idx})

    # Vocab
    vocabs = build_qa_vocab(data)
    all_tokens = set({token for token in vocabs if vocabs[token] >= min_freq})
    all_tokens.update(extra_vocab_tokens)
    vocab2id = dict(zip(all_tokens, range(len(RESERVED_TOKENS), len(all_tokens) + len(RESERVED_TOKENS))))
    for token, idx in RESERVED_TOKENS.items():
        vocab2id.update({token: idx})

    print('Num of entities: %s' % len(entity2id))
    print('Num of entity_types: %s' % len(entityType2id))
    print('Num of relations: %s' % len(relation2id))
    print('Num of vocabs: %s' % len(vocab2id))
    return entity2id, entityType2id, relation2id, vocab2id

def build_ans_cands(graph, entity2id, entityType2id, relation2id):
    cand_ans_entities = [] # answer entity
    cand_ans_types = [] # type of answer entity
    cand_ans_paths = [] # relation path from topic entity to answer entity
    cand_ans_ctx = [] # context (i.e., 1-hop entity bows and relation bows) connects to the answer path
    cand_labels = [] # candidiate answers

    if len(cand_labels) == 0 and (not 'neighbors' in graph or len(graph['neighbors']) == 0):
        return ([], [], [], [], [])

    for k, v in graph['neighbors'].items():
        if if_filterout(k):
            continue
        for nbr in v:
            if isinstance(nbr, str):
                cand_ans_entities.append(RESERVED_ENTS['PAD'])
                cand_ans_types.append([])
                cand_ans_paths.append([relation2id[k] if k in relation2id else RESERVED_RELS['UNK']])
                cand_ans_ctx.append([[], []])
                cand_labels.append(nbr)
                continue
            elif isinstance(nbr, bool):
                cand_ans_entities.append(RESERVED_ENTS['PAD'])
                cand_ans_types.append([entityType2id['bool']])
                cand_ans_paths.append([relation2id[k] if k in relation2id else RESERVED_RELS['UNK']])
                cand_ans_ctx.append([[], []])
                cand_labels.append('true' if nbr else 'false')
                continue
            elif isinstance(nbr, float):
                cand_ans_entities.append(RESERVED_ENTS['PAD'])
                cand_ans_types.append([entityType2id['num']])
                cand_ans_paths.append([relation2id[k] if k in relation2id else RESERVED_RELS['UNK']])
                cand_ans_ctx.append([[], []])
                cand_labels.append(str(nbr))
                continue
            elif isinstance(nbr, dict):
                nbr_k = list(nbr.keys())[0]
                nbr_v = nbr[nbr_k]
                selected_names = (nbr_v['name'] + nbr_v['alias'])[:1]
                is_dummy = True
                if not IGNORE_DUMMY or len(selected_names) > 0: # O therwise, it is an intermediate (dummpy) node
                    cand_ans_entities.append(entity2id[nbr_k] if nbr_k in entity2id else RESERVED_ENTS['UNK'])
                    selected_types = (nbr_v['notable_types'] + nbr_v['type'])[:ENT_TYPE_HOP]
                    cand_ans_types.append([entityType2id[x] if x in entityType2id else RESERVED_ENT_TYPES['UNK'] for x in selected_types])
                    cand_ans_paths.append([relation2id[k] if k in relation2id else RESERVED_RELS['UNK']])
                    cand_labels.append(selected_names[0] if len(selected_names) > 0 else 'UNK')
                    is_dummy = False

                if not 'neighbors' in nbr_v:
                    if not is_dummy:
                        cand_ans_ctx.append([[], []])
                    continue

                
                labels = []
                all_ctx = [set(), set()]
                all_ctx_id = set()
                for kk, vv in nbr_v['neighbors'].items(): # 2nd hop
                    if if_filterout(kk):
                        continue
                    all_ctx[1].add(kk)
                    for nbr_nbr in vv:
                        if isinstance(nbr_nbr, str):
                            cand_ans_entities.append(RESERVED_ENTS['PAD'])
                            cand_ans_types.append([])
                            cand_ans_paths.append([relation2id[k] if k in relation2id else RESERVED_RELS['UNK'], relation2id[kk] if kk in relation2id else RESERVED_RELS['UNK']])
                            labels.append(nbr_nbr)
                            all_ctx[0].add(nbr_nbr)
                            all_ctx_id.add(RESERVED_ENTS['PAD'])
                            continue
                        elif isinstance(nbr_nbr, bool):
                            cand_ans_entities.append(RESERVED_ENTS['PAD'])
                            cand_ans_types.append([entityType2id['bool']])
                            cand_ans_paths.append([relation2id[k] if k in relation2id else RESERVED_RELS['UNK'], relation2id[kk] if kk in relation2id else RESERVED_RELS['UNK']])
                            labels.append('true' if nbr_nbr else 'false')
                            all_ctx[0].add('true' if nbr_nbr else 'false')
                            all_ctx_id.add(RESERVED_ENTS['PAD'])
                            continue
                        elif isinstance(nbr_nbr, float):
                            cand_ans_entities.append(RESERVED_ENTS['PAD'])
                            cand_ans_types.append([entityType2id['num']])
                            cand_ans_paths.append([relation2id[k] if k in relation2id else RESERVED_RELS['UNK'], relation2id[kk] if kk in relation2id else RESERVED_RELS['UNK']])
                            labels.append(str(nbr_nbr))
                            all_ctx[0].add(str(nbr_nbr))
                            all_ctx_id.add(RESERVED_ENTS['PAD'])
                            continue
                        elif isinstance(nbr_nbr, dict):
                            nbr_nbr_k = list(nbr_nbr.keys())[0]
                            nbr_nbr_v = nbr_nbr[nbr_nbr_k]
                            selected_names = (nbr_nbr_v['name'] + nbr_nbr_v['alias'])[:1]
                            if not IGNORE_DUMMY or len(selected_names) > 0:                                
                                cand_ans_entities.append(entity2id[nbr_nbr_k] if nbr_nbr_k in entity2id else RESERVED_ENTS['UNK'])
                                selected_types = (nbr_nbr_v['notable_types'] + nbr_nbr_v['type'])[:ENT_TYPE_HOP]
                                cand_ans_types.append([entityType2id[x] if x in entityType2id else RESERVED_ENT_TYPES['UNK'] for x in selected_types])
                                cand_ans_paths.append([relation2id[k] if k in relation2id else RESERVED_RELS['UNK'], relation2id[kk] if kk in relation2id else RESERVED_RELS['UNK']])
                                labels.append(selected_names[0] if len(selected_names) > 0 else 'UNK')
                                if len(selected_names) > 0:
                                    all_ctx[0].add(selected_names[0])
                                all_ctx_id.add(entity2id[nbr_nbr_k] if nbr_nbr_k in entity2id else RESERVED_ENTS['UNK'])
                        else:
                            raise RuntimeError('Unknown type: %s' % type(nbr_nbr))
                if not is_dummy:
                    ctx_ent = [x for x in all_ctx_id]
                    #cand_ans_ctx.append([ctx_ent])
                    cand_ans_ctx.append([ctx_ent])
                cand_labels.extend(labels)
            else:
                raise RuntimeError('Unknown type: %s' % type(nbr))

    return (cand_ans_entities, cand_labels, cand_ans_types, cand_ans_paths, cand_ans_ctx)

def normalize_answer(s):
    """Lower text and remove extra whitespace."""
    def remove_articles(text):
        return re_art.sub(' ', text)

    def remove_punc(text):
        return re_punc.sub(' ', text)  # convert punctuation to spaces

    def white_space_fix(text):
        return ' '.join(text.split())

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))

def mask_question(query, topic_ent, ent_types):
    query = utils.tokenize(query.lower())
    if topic_ent == '':
        return query, None

    # 'NE'               
    ret = process.extract(topic_ent.replace('_', ' '), set(list(zip(*ent_types))[0]), scorer=fuzz.token_sort_ratio)
    if len(ret) == 0:
        return query, None

    topic_men = ret[-1][0]
    topic_tokens = utils.tokenize(topic_men.lower())
    indices = [i for i, x in enumerate(query) if x == topic_tokens[0]]
    for i in indices:
        if query[i: i + len(topic_tokens)] == topic_tokens:
            start_idx = i
            end_idx = i + len(topic_tokens)
            break

    query_template = query[:start_idx] + ['NE'] + query[end_idx:]
    return query_template, topic_men

def build_data(qa, kb, entity2id, entityType2id, relation2id, vocab2id):
    questions_ids = []
    raw_questions = []
    topic_entities = []
    cand_ans_inds = []
    cand_ans_labels = [] 
    gold_ans_inds = []
    gold_ans_labels = [] 
    
    
    for qid, each in enumerate(qa):
        freebase_key = each['freebaseKey']
        if isinstance(freebase_key, list):
            freebase_key = freebase_key[0] if len(freebase_key) > 0 else ''
        
        question, topic_entity = mask_question(each['qText'], freebase_key, each['entities'])
        q = [vocab2id[x] if x in vocab2id else RESERVED_TOKENS['UNK'] for x in question]
        
        questions_ids.append(q)
        raw_questions.append(question)
        topic_entities.append(topic_entity)
        gold_ans_labels.append(each['answers'])

        cand_ans_entities, cand_labels, cand_ans_types, cand_ans_paths, cand_ans_ctx  = build_ans_cands(kb[freebase_key], entity2id, entityType2id, relation2id)
        
        cand_ans_labels.append(cand_labels)
        
        norm_cand_labels = [normalize_answer(x) for x in cand_labels]
        tmp_cand_inds = []
        for a in each['answers']:
            a = normalize_answer(a)
            # Find all the candidiate answers which match the gold answer.
            inds = [i for i, j in zip(count(), norm_cand_labels) if j == a]
            tmp_cand_inds.extend(inds)

        gold_ans_inds.append(tmp_cand_inds)

    print('Num of question: %s' % len(raw_questions))
    print('Num of topic entities: %s' % len(topic_entities))
    print('Num of candidate answer: %s' % len(cand_ans_labels))
    print('Num of ground truth answer: %s' % len(gold_ans_labels))
    
    return (questions_ids, raw_questions, topic_entities, cand_ans_inds, cand_ans_labels, gold_ans_inds, gold_ans_labels)


if __name__ == '__main__':
    args = parser.parse_args()
    topn = args.topn
    

    # path settting
    proprocess_dir = os.path.join(args.dataset_dir, 'preprocess')
    train_path = os.path.join(args.dataset_dir, 'raw_train.json')
    test_path = os.path.join(args.dataset_dir, 'raw_test.json')
    valid_path = os.path.join(args.dataset_dir, 'raw_valid.json')
    KB_path  = os.path.join(args.dataset_dir, 'KB/freebase.json')
    
    entity2id_path = os.path.join(args.dataset_dir, 'preprocess/entity2id.json')
    entityType2id_path = os.path.join(args.dataset_dir, 'preprocess/entityType2id.json')
    relation2id_path = os.path.join(args.dataset_dir, 'preprocess/relation2id.json')
    vocab2id_path = os.path.join(args.dataset_dir, 'preprocess/vocab2id.json')

    if not os.path.exists(proprocess_dir):
        os.makedirs(proprocess_dir)

    # load dataset
    train_data = utils.load_ndjson(train_path)
    test_data = utils.load_ndjson(test_path)
    valid_data = utils.load_ndjson(valid_path)
    freebase = utils.load_ndjson(KB_path, return_type='dict')
    '''
    Dataset: 
    {
        "answers": ["Padm Amidala"], 
        "entities": [["natalie portman", "PERSON"], ["natalie portman", "NP"], ["star wars", "NP"]], 
        "qText": "what character did natalie portman play in star wars?", 
        "qId": "wqr000001", 
        "freebaseKey": "natalie_portman", 
        "freebaseKeyCands": ["natalie_portman", "star_wars",...], 
        "dep_path": []
    }
    KB:
    "algerian_cup": {
        "name": ["Algerian Cup"], 
        "alias": [],
        "notable_types": [],  // only one
        "type": ["/common/topic"], 
        "neighbors": { 
             "/location/hud_county_place/place":[
                 // maybe string
             {
                "/m/0sfcg":{
                   "name":[
                      "West Peoria"
                   ],
                   "alias":[
                      "West Peoria, Illinois",
                      "Peoria County / West Peoria city"
                   ],
                   "notable_types":[
                      "/location/citytown"
                   ],
                   "type":[
                      "/location/citytown",
                   ]
                }
             }
          ],
        }, 
        "id": "/m/06md5w"
    }
    '''
    
    # build id
    used_fbkeys = set()
    for each in train_data + valid_data:
        used_fbkeys.update(each['freebaseKeyCands'][:topn])

    print('# of used_fbkeys: {}'.format(len(used_fbkeys)))
    
    entity2id, entityType2id, relation2id, vocab2id = build_vocab(train_data + valid_data, freebase, used_fbkeys)
    #utils.dump_json(entity2id, entity2id_path)
    #utils.dump_json(entityType2id, entityType2id_path)
    #utils.dump_json(relation2id, relation2id_path)
    #utils.dump_json(vocab2id, vocab2id_path)
   

    # preprocess data
    train_dataset = build_data(train_data, freebase, entity2id, entityType2id, relation2id, vocab2id)
    valid_dataset = build_data(valid_data, freebase, entity2id, entityType2id, relation2id, vocab2id)
    test_dataset= build_data(test_data, freebase, entity2id, entityType2id, relation2id, vocab2id)
    
    # dump_json(train_dataset, os.path.join(args.out_dir, 'train_vec.json'))
    # dump_json(valid_dataset, os.path.join(args.out_dir, 'valid_vec.json'))
    # dump_json(test_dataset, os.path.join(args.out_dir, 'test_vec.json'))

    # build_utils.mark_done(args.out_dir)
    