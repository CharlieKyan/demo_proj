import sys
sys.path.append('../')
from ans_punct import prep_ans
import json

DATASET_PATH = 'implementation/datasets/'
ANSWER_PATH = {
    'slake_train': DATASET_PATH + 'slake/train.json',
    'slake_val': DATASET_PATH + 'slake/validate.json',
}


train_ds = json.load(open(ANSWER_PATH['slake_train'], 'r', encoding= 'utf-8'))
val_ds = json.load(open(ANSWER_PATH['slake_val'], 'r', encoding= 'utf-8'))

# train_ds_eng = [entry for entry in train_ds if entry['q_lang'] == 'en']
# val_ds_eng = [entry for entry in val_ds if entry['q_lang'] == 'en']

# Loading answer word list
stat_ans_list = []
for data in [train_ds, val_ds]:
    for entry in data:
        stat_ans_list.append(prep_ans(entry['answer']))

print(f'== Total answer list: {len(stat_ans_list)}')

def ans_stat(stat_ans_list):
    ans_to_ix = {}
    ix_to_ans = {}

    for ans in stat_ans_list:
        ix_to_ans[ans_to_ix.__len__()] = ans
        ans_to_ix[ans] = ans_to_ix.__len__()

    return ans_to_ix, ix_to_ans

ans_to_ix, ix_to_ans = ans_stat(stat_ans_list)
print(ans_to_ix.__len__())

json.dump([ans_to_ix, ix_to_ans], open('implementation/datasets/slake/answer_dict.json', 'w'),ensure_ascii=False, indent=4)