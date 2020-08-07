import re
import os
import javalang
import json

def is_identifier(token):
    if re.match(r'\w+', token) and not re.match(r'\d+', token):
        if token not in javalang.tokenizer.Keyword.VALUES.union(javalang.tokenizer.BasicType.VALUES)\
                .union(javalang.tokenizer.Modifier.VALUES):
            return True
    return False

def get_try_index(code):
    start = 0
    stack = []
    for i, token in enumerate(code.split()):
        if token == 'try' and not stack:
            start = i
        elif token in ["'", '"']:
            if stack:
                if stack[-1] == token:
                    stack.pop()
            else:
                stack.append(token)
    return start

def get_statements(code):
    tokens = code.split() if isinstance(code, str) else code
    intervals = []
    stack = []
    start = 0
    flag = False
    for i, token in enumerate(tokens):
        if token in ['"', "'"]:
            if stack:
                if stack[-1] == token:
                    stack.pop()
            else:
                stack.append(token)
            continue

        if not stack:
            if token in ['{', '}', ';'] and not flag:
                intervals.append((start, i))
                start = i+1
            elif token == '(':
                flag = True
            elif token == ')':
                flag = False

    statements = [(tokens[item[0]: item[1]+1], item) for item in intervals]
    return statements


def slicing_mask(front, back):
    tokens = back
    seeds = set()
    for i, token in enumerate(tokens):
        if is_identifier(token):
            if tokens[i+1] != '(' and not is_identifier(tokens[i+1]):
                seeds.add(token)

    tokens = front
    statements = get_statements(tokens)

    st_list = []

    for n, st in enumerate(reversed(statements)):
        flag = False
        assignment_flag = False
        depend = False
        for i, token in enumerate(st[0]):
            if token is '=':
                flag = True

            if is_identifier(token) and not flag and token in seeds:
                depend = True
                assignment_flag = True
                continue
            if assignment_flag and flag:
                try:
                    if is_identifier(token) and tokens[i+1] != '(':
                        seeds.add(token)
                except IndexError:
                    pass
        if depend:
            st_list.append(st[1])
    method_def = statements[0][1]
    if method_def not in st_list:
        st_list.append(method_def)

    code = ' '.join(front)+' '+' '.join(back)
    mask = [0]*len(front)
    for item in st_list:
        mask[item[0]:item[1]] = [1]*(item[1]-item[0])
    assert sum(mask) > 1 and len(front) == len(mask), print(code)

    return ' '.join(front), ' '.join(back), mask

def mask_slicing(dataset):
    origin_root = 'data/baseline/'
    with open(origin_root+'src-%s.txt'%dataset) as fps, open(origin_root+'tgt-%s.txt'%dataset) as fpt:
        origin_src = fps.readlines()
        origin_tgt = fpt.readlines()

    target_root = 'data/multi_slicing/'
    os.makedirs(target_root, exist_ok=True)
    with open(target_root+'src-%s.front'%dataset, 'w') as fwf, open(target_root+'src-%s.back'%dataset, 'w') as fwb,\
            open(target_root+'src-%s.mask'%dataset, 'w') as fwm, \
            open(target_root+'tgt-%s.txt'%dataset, 'w') as fwt:
        for i, (s, t) in enumerate(zip(origin_src, origin_tgt)):
            print(i)
            s = s.strip()
            if not re.match(r'\w+', s):
                print(s)
                s = re.sub(r'^.*?(\w+)', r' \1', s)
                print(s)
            s = re.sub(r'\\\\', ' ', s)
            s = re.sub(r'\\ "', ' \\"', s)
            try_idx = get_try_index(s)
            if not try_idx:
                print('try not found: ', s)
                exit(-1)
            s = s.split()
            front = s[:try_idx]
            back = s[try_idx:]
            front, back, mask = slicing_mask(front, back)
            mask = json.dumps(mask)
            fwf.write(front+'\n')
            fwb.write(back+'\n')
            fwm.write(mask+'\n')
            fwt.write(t)

mask_slicing('train')
mask_slicing('valid')
mask_slicing('test')
