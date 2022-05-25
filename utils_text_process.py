import sys
import re
import string
from collections import Counter
# from sklearn.metrics import classification_report
from seqeval.metrics import classification_report
import spacy
import string

def token_label(textfile):
    text = textfile.readlines()
    
    texts = []
    for line in text:
        if line == '\n':
            texts.append(line)
        else:
            texts.append(line.split(' ')[0] + '\t' + line.split(' ')[3])
            
    return texts

def merge_docs(file1, file2):
    text1 = file1.readlines()
    text2 = file2.readlines()
    
    for line in text2:
        text1.append(line)
        
    return text1

def find_majority(votes):
    vote_count = Counter(votes)
    top_two = vote_count.most_common(2)
    if len(top_two)>1 and top_two[0][1] == top_two[1][1]:
        # It is a tie
        # return "TIE"
        return "O"
    return top_two[0][0]

def extr_column(textfile, col):
    # col is the column number starts from 0
    text = textfile.readlines()
    
    texts = []
    for line in text:
        if line == '\n':
            texts.append(line)
        else:
            texts.append(line.split('\t')[col])
            
    return texts

def write_data(text, output):
    with open(output,'w', encoding='utf-8') as fout:
        for line in text:
            fout.writelines(line)

def split_set(ner_file):
    ner_text = ner_file.read().split('\n\n')

    train = []
    for sent in ner_text[:14986]:
        train.append(sent)
        train.append('\n\n')

    dev = []
    for sent in ner_text[14986:18452]:
        dev.append(sent)
        dev.append('\n\n')

    test = []
    for sent in ner_text[18452:]:
        test.append(sent)
        test.append('\n\n')

    return train, dev, test

def took_some(ner_file, num_sentence):
    ner_text = ner_file.read().split('\n\n')
    len_nertext = len(ner_text)

    took = []
    for sent in ner_text[:num_sentence]:
        took.append(sent)
        took.append('\n\n')
    len_took = len(took)

    return len_nertext, len_took, took

def tok2sent(textfile):
    # TreebankWordDetokenizer().detokenize(['the', 'quick', 'brown'])
    # 'The quick brown'
    text = textfile.readlines()
    
    sentence = []
    for line in text:
        if line == '\n':
            sentence.append(line)
        else:
            sentence.append(line.split()[0] + ' ')

    return sentence

def merge_token_label(token_file, label_file):
    tokens = token_file.readlines()
    labels = label_file.readlines()

    sentences = []
    for token, label in zip(tokens, labels):
        token = token.split()
        label = label.split()
        
        sent = []
        for t, l in zip(token, label):
            merged = t + '\t' + l + '\n'
            sent.append(merged)
        
        sentences.append(sent)
        sentences.append('\n')
    
    return sentences

def label2sent(textfile):
    text = textfile.readlines()
    
    sentence = []
    for line in text:
        if line == '\n':
            sentence.append(line)
        else:
            sentence.append(line.split()[1] + ' ')

    return sentence

def extract_labels(textfile):
    text = textfile.readlines()
    
    labels = []
    for line in text:
        if line == '\n':
            labels.append(line)
        else:
            l = line.split()
            labels.append(l[1])

    return labels

def ner_tagger_wiki(textfile):
    text = textfile.readlines()
    nlp = spacy.load("xx_ent_wiki_sm")

    sentences = []
    for line in text:
        doc = nlp(line)

        token = []
        iob = []
        tag = []
        for ent in doc:
            token.append(ent.text)
            iob.append(ent.ent_iob_)
            tag.append(ent.ent_type_)


        tokens = []
        for t, b, ta in zip(token[:-1], iob[:-1], tag[:-1]):
            if b == 'O':
                tokens.append(t + '\t' + b)
            else:
                tokens.append(t + '\t' + b + '-' + ta)
        
        sentences.append('\n'.join(tokens))
        sentences.append('\n\n')

    return sentences

def ner_tagger_core_spacy(textfile):
    text = textfile.readlines()
    nlp = spacy.load("en_core_web_sm")

    sentences = []
    for line in text:
        doc = nlp(line)

        token = []
        iob = []
        tag = []
        for ent in doc:
            token.append(ent.text)
            iob.append(ent.ent_iob_)
            tag.append(ent.ent_type_)


        tokens = []
        for t, b, ta in zip(token[:-1], iob[:-1], tag[:-1]):
            if ta == 'PERSON':
                tokens.append(t + '\t' + b + '-' + ta[:3])
            elif ta == 'LOC' or ta == 'ORG':
                tokens.append(t + '\t' + b + '-' + ta)
            elif ta == 'GPE':
                tokens.append(t + '\t' + b + '-' + 'ORG')
            else:
                tokens.append(t + '\t' + 'O')
        
        sentences.append('\n'.join(tokens))
        sentences.append('\n\n')

    return sentences

# def evaluation(gold_file, pred_file):
#     gold = gold_file.readlines()
#     pred = pred_file.readlines()

#     report = classification_report(gold, pred)

#     return report

def evaluation(gold_file, pred_file):
    gold_f = gold_file.read().split('\n\n')
    pred_f = pred_file.read().split('\n\n')
    
    gold = []
    for line in gold_f:
        line_n = line.split()
        toks = []
        for tok in line_n:
            toks.append(tok)
        gold.append(toks)

    pred = []
    for line in pred_f:
        line_n = line.split()
        toks = []
        for tok in line_n:
            toks.append(tok)
        pred.append(toks)

    # print(gold[:2])
    report = classification_report(gold, pred, digits=4)

    return report

def read_giza(filetext):
    text = filetext.readlines()

    sources = []
    targets = []
    for idx, line in enumerate(text):
        if idx % 3 == 1:
            sources.append(line)
        elif idx % 3 == 2:
            targets.append(line)
        
    # line_1 = "Menurut media Iran yang dijalankan negara , presiden Iran sekarang Mahmoud Ahmadinejad telah memenangkan pemilihan ulang dengan selisih 2 hingga 1 . "
    # line_2 = "NULL ({ 4 }) According ({ 1 2 }) to ({ }) Iranian ({ 3 5 6 7 8 }) state ({ 9 }) run ({ 10 }) media ({ 11 }) , ({ }) current ({ 12 }) Iranian ({ }) president ({ 13 }) Mahmoud ({ 14 }) Ahmadinejad ({ }) has ({ }) won ({ 15 }) re ({ 16 }) - ({ 17 }) election ({ 18 }) by ({ }) a ({ }) margin ({ 19 }) of ({ }) 2 ({ 20 }) to ({ }) 1 ({ 21 }) . ({ 22 }) "
    docs = []
    for src, tgt in zip(sources, targets):
        token_list_1 = src.split()
        token_list_2 = tgt.split()
        aligned = ''
        flag = True
        correspond_token_list = []
        for token in token_list_2:
            if True == flag:
                if '({' == token:
                    flag = False
                    aligned += ' '
                    correspond_token_list = []
                else:
                    aligned += token + ' '
            else:
                if '})' == token:
                    flag = True
                    aligned += ','.join(correspond_token_list) + '\n'
                else:
                    correspond_token_list.append(token_list_1[int(token)-1])

        punct = set(string.punctuation)
        sents = []
        aligned_tok = aligned.split('\n')
        for token in aligned_tok[:-1]:
            pairs = token.split()
            if len(pairs) == 2:
                if pairs[0] != 'NULL':
                    if any (char in punct for char in pairs[1]):
                        if len(pairs[1].split(',')) > 1:
                            sents.append(pairs[0] + ' ' + pairs[1].split(',')[0] + '\n')
                        else:
                            sents.append(' '.join(pairs) + '\n')
                    else:
                        sents.append(pairs[0] + ' ' + pairs[1] + '\n')
        
        docs.append(sents)
        docs.append('\n')

    return docs


def src_tgt(src_file, tgt_file):
    src = src_file.readlines()
    tgt = tgt_file.readlines()

    merged = []
    for s, t in zip(src, tgt):
        merged.append(s[:-1] + '||| ' + t)

    return merged

def read_align(src_file, tgt_file, align_file):
    src = src_file.readlines()
    tgt = tgt_file.readlines()
    alg = align_file.readlines()

    # src = "Of those players charged , 27 were found guilty of at least one charge ."
    # src_label = "O O O O O O O O O O O O O O O "
    # tgt = "Pemain yang dinyatakan didakwa , 27 dinyatakan bersalah pada sedikitnya satu tuduhan ."
    # alg = "1-0 2-1 2-2 3-3 4-4 5-5 6-6 7-6 8-7 10-8 11-9 12-10 13-11 13-12"

    docs_aligned = []
    for src_l, tgt_l, alg_l in zip(src, tgt, alg):
        token_src = src_l.split()
        token_tgt = tgt_l.split()
        token_alg = alg_l.split()

        for al in token_alg:
            pair = al.split('-')
            sent = []
            for i, s in enumerate(token_src):
                for j, t in enumerate(token_tgt):
                    if i == int(pair[0]) and j == int(pair[1]):
                        sent.append(t + '\t' + s + '\n')
            docs_aligned.append(sent)
        docs_aligned.append('\n')

    return docs_aligned

def alignment2ner_old(src_label_file, tgt_file, align_file):
    src_label = src_label_file.readlines()
    tgt = tgt_file.readlines()
    alg = align_file.readlines()

    docs_temp = []
    for src_label_l, tgt_l, alg_l in zip(src_label, tgt, alg):
        token_src_label = src_label_l.split()
        token_tgt = tgt_l.split()
        token_alg = alg_l.split()

        for al in token_alg:
            pair = al.split('-')
            sent = []
            for i, s_l in enumerate(token_src_label):
                for j, t in enumerate(token_tgt):
                    if i == int(pair[0]) and j == int(pair[1]):
                        sent.append(t + '\t' + s_l + '\n')
            docs_temp.append(sent)
        docs_temp.append('\n')
    
    docs = []
    for curr_line, next_line in zip(docs_temp, docs_temp[1:]):
        if next_line != curr_line:
            docs.append(curr_line)

    return docs

def alignment2ner(src_label_file, tgt_file, align_file):
    src_label = src_label_file.readlines()
    tgt = tgt_file.readlines()
    alg = align_file.readlines()

    docs_temp = []
    for src_label_l, tgt_l, alg_l in zip(src_label, tgt, alg):
        token_src_label = src_label_l.split()
        token_tgt = tgt_l.split()
        token_alg = alg_l.split()

        for al in token_alg:
            pair = al.split('-')
            sent = []
            for i, s_l in enumerate(token_src_label):
                for j, t in enumerate(token_tgt):
                    if i == int(pair[0]) and j == int(pair[1]):
                        sent.append(t + '\t' + s_l + '\n')
            docs_temp.append(sent)
        docs_temp.append('\n')
    
    docs = []
    for curr_line, next_line in zip(docs_temp, docs_temp[1:]):
        for c, n in zip(curr_line, next_line):
            curr_tok = c.split('\t')[0]
            next_tok = n.split('\t')[0]
            if next_tok != curr_tok:
                docs.append(next_line)

    return docs

def giza2ner(giza_outfile, nerfile):
    giza = giza_outfile.read().split('\n\n')
    ner = nerfile.read().split('\n\n')

    docs_temp = []
    for giza_tokens, ner_tokens in zip(giza, ner):
        giza_tokens = giza_tokens.split('\n')
        ner_tokens = ner_tokens.split('\n')
        sent_temp = []
        for g in giza_tokens:
            for n in ner_tokens:
                if len(g.split(' ')) == 2:
                    id, en = g.split(' ')
                    tok, lab = n.split('\t')
                    if id != 'NULL':
                        if en == tok:
                            sent_temp.append(id + ' ' + en + ' ' + lab)
        
        docs_temp.append(sent_temp)
        # docs_temp.append('\n')
    
    docs = []
    for s_temp in docs_temp: 
        sent = []
        for curr_line, next_line in zip(s_temp, s_temp[1:]):
            curr_tok = curr_line.split(' ')[0]
            next_tok = next_line.split(' ')[0]
            if next_tok != curr_tok:
                sent.append(curr_line.split()[0] + ' ' + curr_line.split()[2] + '\n')
        docs.append(sent)
        docs.append('\n')

    return docs

def remove_nonentity(ner_file):
    ner_text = ner_file.read().split('\n\n')

    docs = []
    for sent in ner_text:
        line = sent.splitlines()

        for tokens in line:
            tok, lab = tokens.split(' ')
            if lab != 'O':
                if sent not in docs:
                    docs.append(sent)
                    docs.append('\n\n') 
        # print(tokens + 'X')
    return docs   

def token2prompt(nerfile):
    ner = nerfile.read().split('\n\n')

    nerLabelEN = { # English Label
        "PER": "person",
        "ORG": "organization",
        "LOC": "location",
        "MIS": "other"
    }

    nerLabelID = { # Indonesian Label
        "PER": "tokoh",
        "ORG": "organisasi",
        "LOC": "lokasi",
        "PPL": "tokoh",
        "FNB": "makanan",
        "PLC": "lokasi",
        "EVT": "event",
        "IND": "produk"
    }

    for sent in ner:
        line = sent.split('\n')
        ent = []
        prevLabel = ''
        prevToken = ''
        sentence = []
        sent_ent = []
        for tokens in line:
            # print(tokens)
            tok, lbl = tokens.split('\t')
            # print(tok)
            sentence.append(tok)
            # print(tok,lbl, prevLabel)

            # # with the non-entities
            # if prevLabel == 'O' and len(prevToken) == 1:
            #     ent = [tok]
            # elif prevLabel == 'O' and len(prevToken) > 1:
            #     # print(ent[0] + ' is not an entity.')
            #     if prevLabel not in ['']:
            #         # print(' '.join(ent) + ' is a ' + nerLabel[prevLabel] +' entity.')
            #         # sent_ent.append(' '.join(ent) + ' is a not an entity') # template in english
            #         sent_ent.append(' '.join(ent) + ' adalah bukan entitas .') # template in indonesian
            #     ent = [tok]

            # without the non-entities   
            if prevLabel == 'O':
                ent = [tok]
            else:
                if prevLabel == lbl[-3:]:
                    ent.append(tok)
                else:
                    # print('-->', prevLabel)
                    if prevLabel not in ['']:
                        # print(' '.join(ent) + ' is a ' + nerLabel[prevLabel] +' entity.')
                        # sent_ent.append(' '.join(ent) + ' is a ' + nerLabel[prevLabel] +' entity') # template in english
                        sent_ent.append(' '.join(ent) + ' adalah entitas ' + nerLabelID[prevLabel] + ' .') # template in indonesian
                    ent = [tok]                            
            prevLabel = lbl[-3:]
            prevToken = tok
        for i in sent_ent:
            print('"' + ' '.join(sentence) + '",' + i)
            # save the output using cli > blablabla.out

def lowercased(textfile):
    file = textfile.readlines()

    lowercased_text = []
    for line in file:
        lowercased_line = line.lower()
        # print(lowercased_line)
        lowercased_text.append(lowercased_line)
    
    return lowercased_text

def remove_line_cond(condition, filename):
    for line in filename:
        if not condition in line:
            print(line[:-1])

if __name__ == "__main__":
    # # Extract token_label data
    # text = open('./data/test_wpos.txt', 'r', encoding='utf-8')
    # extracted = token_label(text)
    # write_data(extracted, './data/test-.txt')

    # Convert BIO to Prompt
    file = open('./data/conll03/dev_en_trans_idn.csv')
    textfile = lowercased(file)
    # textfile = token2prompt(file)
    # print(textfile)

    write_data(textfile, './data/conll03/dev_en_trans_idn_low.csv')

    # # Took first x sentence from NER file
    # text = open('./data/modeltf/en-train.txt', 'r', encoding='utf-8')
    # len_text, len_ext, extracted = took_some(text, 8500)
    # print(len_text)
    # print(len_ext)
    # write_data(extracted, './data/modeltf/en-train-8k.txt')