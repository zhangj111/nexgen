import pandas as pd
import re

data = pd.read_csv('exp_data.csv')
# NUMBER = 9
# print(data['code'][NUMBER])
# split_code_line(data['code'][NUMBER])


def filter_try_catch(source):
    source = str(source)
    if len(re.findall(r'\Wcatch\W', source)) == 0 and len(re.findall(r'\Wtry\W', source)) == 1:
        return False
    return True


def split_code_line(source):
    source = str(source)
    source = re.sub(r'\Wtry\W\s*{', '###start###', source)
    source = re.sub(r'}\s*catch\W[\s\S]*?{[\s\S]*?}', '###end###', source)
    # print(source)
    lines = re.split('\n+', source)
    results = []
    labels = []
    flag = False
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if line == '###start###':
            flag = True
        elif line == '###end###':
            flag = False
        else:
            results.append(line)
            if flag:
                labels.append(1)
            else:
                labels.append(0)
    # print(results)
    # print(labels)
    assert len(results) == len(labels)
    return results, labels


format_data = data['code'].apply(split_code_line)
task1_data = pd.DataFrame(data=format_data.tolist(), columns=['lines', 'labels'])
task1_data.to_pickle('data/task1_data.pkl')




