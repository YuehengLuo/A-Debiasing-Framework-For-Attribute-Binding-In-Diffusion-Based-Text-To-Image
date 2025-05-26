import torch
from transformers import pipeline
import stanza
from nltk.tree import Tree, ParentedTree
import re
import numpy as np


skip_nouns = ["photo", "bunches", "bunch", "front", "patch", "side",
              "pile", "piece"]



def check_prompt(text: str) -> str:
    # 使用正则表达式在标点符号前后添加空格
    text = re.sub(r'([,.\'!?;":\-])', r' \1 ', text)
    # 将多个连续的空格替换为单个空格
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()


def modify_prompt(text: str, replaced_span, replaced_text):
    if len(replaced_span) == 1:
        replaced_span = [replaced_span[0], replaced_span[0]+1]
    if type(replaced_text) is str:
        replaced_text = [replaced_text]

    split_text = text.split(' ')
    output = split_text[:replaced_span[0]] + replaced_text + split_text[replaced_span[-1]:]
    return ' '.join(output)


def get_span(sentence, sub_sentence, span=np.array([0, 99])):
    list_sentence = sentence.split(' ')
    list_sub = sub_sentence.split(' ')

    output = []
    cur_word = 0
    for i, word in enumerate(list_sentence):
        if word == list_sub[cur_word]:
            if i >= span[0] and i <= span[-1]:
                output.append(i)
                cur_word += 1
            if len(output) == len(list_sub):
                return np.array(output)
        else:
            output = []
            cur_word = 0
    return np.array(output)


def get_word_inds(text: str, word_place, tokenizer, text_inds=None):

    if text_inds is None:
        text_inds = tokenizer.encode(text)

    split_text = text.split()
    if isinstance(word_place, str):
        word_place = [i for i, word in enumerate(split_text) if word_place.lower() == word]
    elif isinstance(word_place, int):
        word_place = [word_place]
    out = []
    if len(word_place) > 0:
        # Decode tokenized words and remove special characters like '#'
        words_encode = [tokenizer.decode([item]).strip("#") for item in text_inds][1:-1]
        cur_len, ptr = 0, 0
        for i in range(len(words_encode)):
            cur_len += len(words_encode[i])
            if ptr in word_place:
                out.append(i + 1)
            if cur_len >= len(split_text[ptr]):
                ptr += 1
                cur_len = 0
    return out


def extract_attribution_indices(doc):
    # doc = parser(prompt)
    subtrees = []
    modifiers = ["amod", "nmod", "compound", "npadvmod", "advmod", "acomp"]

    for w in doc:
        if w.pos_ not in ["NOUN", "PROPN"] or w.dep_ in modifiers:
            continue
        subtree = {}
        stack = []
        attribute = []
        for child in w.children:
            if child.dep_ in modifiers:
                attribute.append(child.text)
                stack.extend(child.children)

        while stack:
            node = stack.pop()
            if node.dep_ in modifiers or node.dep_ == "conj":
                attribute = [node.text] + attribute
                stack.extend(node.children)

        subtree["attribute"] = " ".join(attribute)
        subtree["subject"] = w.text

        subtree["concept"] = " ".join(attribute + [w.text])

        subtrees.append(subtree)

    return subtrees


def get_pairs(text: str, parser=None):
    # 初始化NLP解析器
    if parser is None:
        nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency', download_method=None)
    else:
        nlp = parser

    # 解析文本并获取句子的语法树
    parse_doc = nlp(text)
    tree = Tree.fromstring(str(parse_doc.sentences[0].constituency))
    tree = ParentedTree.convert(tree)

    def extract_pairs(tree):
        pairs = []
        if type(tree) == str or tree is None:
            return []

        # 判断树中的名词节点
        if tree.label() in ['NN', 'NNS'] and tree.leaves()[0] == tree.parent().leaves()[-1]:
            cut_off = 0
            if tree.parent()[0].label() == 'DT':
                cut_off = 1

            # 获取概念词
            concept = ' '.join(tree.parent().leaves()[cut_off:])
            if concept in skip_nouns:
                return []

            # 提取主体和属性，并生成pair
            subject = ' '.join(tree.leaves())
            attribute = ' '.join(concept.split(' ')[:-1])
            
            # 仅当concept和attribute不为空时才添加
            if subject and attribute:
                pairs.append({
                    'subject': subject,
                    'attribute': attribute,
                    'concept': concept,
                })

        # 递归获取子树中的pair
        for subtree in tree:
            pairs += extract_pairs(subtree)

        return pairs

    # 获取所有pair
    all_pairs = extract_pairs(tree)

    # 去除重复或嵌套的属性
    all_concepts = [pair['concept'] for pair in all_pairs]
    all_attributes = [pair['attribute'] for pair in all_pairs]
    remove_list = []

    for p_id, concept in enumerate(all_concepts):
        for attribute in all_attributes:
            if concept in attribute:
                remove_list.append(p_id)

    # 返回有效的pair
    output = []
    for p_id, pair in enumerate(all_pairs):
        if p_id in remove_list:
            continue
        # 仅当pair的concept和attribute都不为空时才添加
        if pair['concept'] and pair['attribute']:
            output.append(pair)

    return output



def get_substitutes(model, masked_text: str, k=10, threshold=0.02):
    if '.' not in masked_text:
        masked_text += '.'

    # masked_text = '[CLS]' + masked_text + '[SEP]'
    substitutes = []
    outputs = model(masked_text, top_k=k)
    # print(outputs)
    for output in outputs:
        if output['score'] > threshold:
            word = output['token_str'].strip('#')
            substitutes.append(word)
    return substitutes


def gather_word_vector(vectors, indices):
    gather_index = torch.tensor(indices).to(vectors.device)
    gather_index = gather_index[..., None, None].expand(-1, -1, vectors.shape[-1])
    output = vectors.gather(1, gather_index)
    return output
