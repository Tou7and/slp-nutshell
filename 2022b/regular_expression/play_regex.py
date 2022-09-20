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

# Unicode range of Mandarin: 4E00-9FFF
zh_1 = re.compile(r"[\u4E00-\u9FFF]") 
zh_2 = re.compile(r"[ㄅ-龜]") # Taiwan people style
zh_3 = re.compile(r"[\u4E00-\u9FFF]|[A-z]+") # Use or to include English words.
zh_4 = re.compile(r"[ㄅ-龜]+|[A-z]+") # Use or to include English words.

def match_mandarin():
    x = "你也是Gundam嗎"
    print(patt_1.findall(x))
    print(zh_1.findall(x))
    print(zh_2.findall(x))
    print(zh_3.findall(x))
    print(zh_4.findall(x))

# Unicode range of Japanese: 4e00-9fbf, 3040-309f and 30a0-30ff
jp_1 = re.compile(r"[\u4e00-\u9fbf]|[\u3040-\u309f]|[\u30a0-\u30ff]") 
jp_2 = re.compile(r"[\u4e00-\u9fbf]+|[\u3040-\u309f]+|[\u30a0-\u30ff]+")
def match_japanese():
    x = "お前もガンダムですか"
    print(patt_1.findall(x))
    print(zh_1.findall(x))
    print(jp_1.findall(x))
    print(jp_2.findall(x))

if __name__ == "__main__":
    # match_english()
    # match_mandarin()
    match_japanese()

