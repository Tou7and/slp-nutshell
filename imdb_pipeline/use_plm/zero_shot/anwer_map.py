import re

def answer_mapping(x):
    # x = "<s> xsasd asdad </t>"
    y = re.sub("<.*?>", "", x).strip()
    # print(y)
    return y
