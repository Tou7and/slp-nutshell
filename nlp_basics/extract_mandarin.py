import re
import unicodedata

def is_chinese_char(cc):
    """ Check if the character is Chinese
    args:
        cc: char
    output:
        boolean
    """
    return unicodedata.category(cc) == 'Lo'

def extract_chinese(text):
    new_text = []
    for char in text:
        if is_chinese_char(char):
            new_text.append(char)
    return "".join(new_text)

def extract_mandarin(text):
    pattern = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]')
    results = pattern.finditer(text)
    zh_chars = []
    for result in results:
        # print(result.group(), result.span())
        zh_chars.append(result.group())
    sent_new = "".join(zh_chars)
    # print(sent_new)
    return sent_new

if __name__ == "__main__":
    with open("bbc_news_0411.txt", 'r') as reader:
        lines = reader.readlines()
    new_line = []
    for line in lines:
        print(extract_mandarin(line))
