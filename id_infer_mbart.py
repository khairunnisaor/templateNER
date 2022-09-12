from utils_metrics import get_entities_bio, f1_score, classification_report
from transformers import MBartForConditionalGeneration, MBart50Tokenizer
from indobenchmark import IndoNLGTokenizer
import torch
import time
import math
import sys

class InputExample():
    def __init__(self, words, labels):
        self.words = words
        self.labels = labels

def template_entity(words, input_TXT, start):
    # input text -> template
    words_length = len(words)
    words_length_list = [len(i) for i in words]

    template_list = [" adalah entitas lokasi .", " adalah entitas orang .", " adalah entitas organisasi .",
                     " adalah bukan entitas ."] # 29/06/2022 change 'tokoh' to 'orang'
    entity_dict = {0: 'LOC', 1: 'PER', 2: 'ORG', 3: 'O'}

    len_temp = len(template_list) 

    # change every 5 to len(template_list)
    input_TXT = [input_TXT] * (len_temp * words_length)
    input_ids = tokenizer(input_TXT, return_tensors='pt')['input_ids']
    model.to(device)

    temp_list = [] # all n-gram are filled in to the templates
    for i in range(words_length):
        for j in range(len_temp):
            temp_list.append(words[i] + template_list[j])

    output_ids = tokenizer(temp_list, return_tensors='pt', padding=True, truncation=True)['input_ids']
    # output_ids[:, 0] = 2 # No need for IndoBART and mBART
    output_length_list = [0] * len_temp * words_length


    for i in range(len(temp_list) // len_temp):
        base_length = ((tokenizer(temp_list[i * len_temp], return_tensors='pt', padding=True, truncation=True)['input_ids']).shape)[1] - 3 # Change to 3 for IndoBART
        output_length_list[i * len_temp : i * len_temp + len_temp] = [base_length] * len_temp
        output_length_list[i * len_temp + (len_temp - 1)] += 1

    score = [1] * len_temp * words_length
    with torch.no_grad():
        output = model(input_ids=input_ids.to(device), decoder_input_ids=output_ids[:, :output_ids.shape[1] - 2].to(device))[0]
        for i in range(output_ids.shape[1] - 3):
            logits = output[:, i, :]
            logits = logits.softmax(dim=1)
            # values, predictions = logits.topk(1,dim = 1)
            logits = logits.to('cpu').numpy()

            for j in range(0, len_temp * words_length):
                if i < output_length_list[j] + 1: # +1 for IndoBART and no effect on mBART but literally no need for BART
                    score[j] = score[j] * logits[j][int(output_ids[j][i + 1])]

    end = start+(score.index(max(score)) // len_temp)
    return [start, end, entity_dict[(score.index(max(score)) % len_temp)], max(score)] #[start_index,end_index,label,score]



def prediction(input_TXT):
    input_TXT_list = input_TXT.split(' ')

    entity_list = []
    for i in range(len(input_TXT_list)):
        words = []
        for j in range(1, min(9, len(input_TXT_list) - i + 1)):
            word = (' ').join(input_TXT_list[i:i+j])
            words.append(word)

        entity = template_entity(words, input_TXT, i) #[start_index,end_index,label,score]
        if entity[1] >= len(input_TXT_list):
            entity[1] = len(input_TXT_list)-1
        if entity[2] != 'O':
            entity_list.append(entity)
    i = 0
    if len(entity_list) > 1:
        while i < len(entity_list):
            j = i+1
            while j < len(entity_list):
                if (entity_list[i][1] < entity_list[j][0]) or (entity_list[i][0] > entity_list[j][1]):
                    j += 1
                else:
                    if entity_list[i][3] < entity_list[j][3]:
                        entity_list[i], entity_list[j] = entity_list[j], entity_list[i]
                        entity_list.pop(j)
                    else:
                        entity_list.pop(j)
            i += 1
    label_list = ['O'] * len(input_TXT_list)

    for entity in entity_list:
        label_list[entity[0]:entity[1]+1] = ["I-"+entity[2]]*(entity[1]-entity[0]+1)
        label_list[entity[0]] = "B-"+entity[2]
    return label_list

def cal_time(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

tokenizer = MBart50Tokenizer.from_pretrained('facebook/mbart-large-50')
model = MBartForConditionalGeneration.from_pretrained(sys.argv[1])

model.eval()
model.config.use_cache = False
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

score_list = []
file_path = './data/idner2k/test_bio.txt'
guid_index = 1
examples = []
with open(file_path, "r", encoding="utf-8") as f:
    words = []
    labels = []
    for line in f:
        if line.startswith("-DOCSTART-") or line == "" or line == "\n":
            if words:
                examples.append(InputExample(words=words, labels=labels))
                words = []
                labels = []
        else:
            splits = line.split("\t") # change the separator following the data (usually tab '\t' or space ' ')
            # print(splits[1])
            words.append(splits[0])
            if len(splits) > 1:
                labels.append(splits[-1].replace("\n", ""))
            else:
                # Examples could have no label for mode = "test"
                # labels.append("O")
                labels.append(splits[1])
    if words:
        examples.append(InputExample(words=words, labels=labels))

trues_list = []
preds_list = []
words_list = []
str = ' '
num_01 = len(examples)
num_point = 0
start = time.time()
for example in examples:
    sources = str.join(example.words)
    words_list.append(example.words)
    preds_list.append(prediction(sources))
    trues_list.append(example.labels)
    num_point += 1


true_entities = get_entities_bio(trues_list)
pred_entities = get_entities_bio(preds_list)
results = {
    "f1": f1_score(true_entities, pred_entities),
    "cr": classification_report(true_entities, pred_entities)
}
# print(results["f1"])
print("Model Name: " + sys.argv[1])
print(results["cr"])
print('\n\n\n\n\n\n\n')

for num_point in range(len(preds_list)):
    words_list[num_point] = ' '.join(words_list[num_point]) + '\n'
    preds_list[num_point] = ' '.join(preds_list[num_point]) + '\n'
    trues_list[num_point] = ' '.join(trues_list[num_point]) + '\n'
with open(sys.argv[2], 'w') as f0:
    for word, gold, pred in zip(words_list, trues_list, preds_list):
        f0.writelines(word)
        f0.writelines(gold)
        f0.writelines(pred)
        f0.writelines('\n')
