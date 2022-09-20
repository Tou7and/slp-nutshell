""" Try Some regex rules """
import re

patt_0 = re.compile(r"(?u)\b\w\w+\b")
patt_1 = re.compile(r"\b\w\w+\b") # equal to patt-0, since (?u) is default.
patt_2 = re.compile(r"\b\w+\b") # Will keep words with at least two characters.
patt_3 = re.compile(r"\b[A-Z]\w+\b") # Will keep only words start with upper cases.

def match_english():
    x = "i eat dinner with John in McDonald last night."
    print(patt_1.findall(x))
    print(patt_2.findall(x))
    print(patt_3.findall(x))

zh_1 = re.compile(r"[\u4E00-\u9FFF]") # CJK Unified Ideographs: 4E00-9FFF
zh_2 = re.compile(r"[ㄅ-龜]")
zh_3 = re.compile(r"[\u4E00-\u9FFF]|[A-z]+") # Use or to include English words.
zh_4 = re.compile(r"[\u4E00-\u9FFF]+|[A-z]+") # Use or to include English words.

def match_mandarin():
    x = "你也是Gundam嗎"
    print(patt_1.findall(x))
    print(zh_1.findall(x))
    print(zh_2.findall(x))
    print(zh_3.findall(x))
    print(zh_4.findall(x))

if __name__ == "__main__":
    # match_english()
    match_mandarin()

