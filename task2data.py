import pandas as pd
import re

data = pd.read_csv('exp_data.csv')
# NUMBER = 1
# print(data['code'][NUMBER])
# cutout_catch(data['code'][NUMBER])


def filter_try_catch(source):
    source = str(source)
    if len(re.findall(r'}\s*catch\W', source)) == 1 and len(re.findall(r'\Wtry\W\s*{', source)) == 1:
        return True
    return False


def cutout_catch(source):
    source = str(source)
    source = re.sub(r'\Wtry\W\s*{', '', source)
    target = re.findall(r'}\s*(catch\W[\s\S]*?{[\s\S]*?})', source)
    if len(target) != 1:
        print(source)
        exit(-1)
    source = re.sub(r'}\s*catch\W[\s\S]*', '', source)
    return source, target[0]


exception_data = data[data['code'].apply(filter_try_catch)]
format_data = exception_data['code'].apply(cutout_catch)
task1_data = pd.DataFrame(data=format_data.tolist(), columns=['source', 'target'])
task1_data.to_pickle('data/task2_data.pkl')


