import numpy as np
import json


def get_corpus(path):
    with open(path) as f:
        line = f.readline()
        a_labels = []
        a_texts = []
        a_intents = []

        s_label = []
        s_text = []
        index = 0
        while line:
            l, r = line[:-1].split(' ')
            if l == 'EOS':
                a_intents.append(r)
                a_labels.append(s_label)
                s_label = []
                a_texts.append(' '.join(s_text))
                s_text.clear()
                index = 0
                f.readline()
            else:
                if r.startswith('B'):
                    s_label.append({
                        'entity': r[2:],
                        'text': l,
                        'start': index,
                        'end': index + len(l)
                    })
                elif r.startswith('I'):
                    s_label[-1]['text'] += ' ' + l
                    s_label[-1]['end'] += 1 + len(l)
                s_text.append(l)
                index += len(l) + 1
            line = f.readline()

    out = []
    for label, text, intent in zip(a_labels, a_texts, a_intents):
        out.append({
            'text': text,
            'intent': intent,
            'entities': label
        })
    return out


def read_train(path):
    with open(path) as f:
        line = f.readline()
        a_labels = []
        a_texts = []
        a_intents = []

        s_label = []
        s_text = []
        index = 0
        while line:
            line = line[:-1]
            text, labels = line.split('\t')
            for l, r in zip(text.split(' '), labels.split(' ')):
                if l == 'EOS':
                    a_intents.append(r)
                    a_labels.append(s_label)
                    s_label = []
                    a_texts.append(' '.join(s_text))
                    s_text.clear()
                    index = 0
                else:
                    if r.startswith('B'):
                        s_label.append({
                            'entity': r[2:],
                            'text': l,
                            'start': index,
                            'end': index + len(l)
                        })
                    elif r.startswith('I'):
                        s_label[-1]['text'] += ' ' + l
                        s_label[-1]['end'] += 1 + len(l)
                    s_text.append(l)
                    index += len(l) + 1
                line = f.readline()
        out = []
        for label, text, intent in zip(a_labels, a_texts, a_intents):
            out.append({
                'text': text,
                'intent': intent,
                'entities': label
            })
    return out


corpora = [
    read_train('./data/atis-2.train.w-intent.iob'),
    read_train('./data/atis.train.w-intent.iob'),
    get_corpus('./data/atis-2.dev.w-intent.iob'),
    get_corpus('./data/atis.test.w-intent.iob')
]

co = []
for t in corpora:
    co.extend(t)

out = {
    'sentences': co
}

with open('ATIS_with_intent.json', 'w') as f:
    json.dump(out, f, indent=2)


