sent = "Steve Jobs手上拿著一隻iphone 好潮ㄛ 褲子都濕ㄌ"

import re
pattern = re.compile('[A-Za-z]+')
results = pattern.finditer(sent)

en_words = []
for result in results:
    print(result.group(), result.span())
    en_words.append(result.group())

print()
sent_new = " ".join(en_words)
print(sent_new)

pattern_zh = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]')
results = pattern_zh.finditer(sent)

zh_chars = []
for result in results:
    print(result.group(), result.span())
    zh_chars.append(result.group())
sent_new = "".join(zh_chars)

print()
print(sent_new)



pattern_zh2 = re.compile(u'[⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎ㄛㄌ]')
results = pattern_zh2.finditer(sent)

zh_chars = []
for result in results:
    print(result.group(), result.span())
    zh_chars.append(result.group())
sent_new = "".join(zh_chars)

print()
print(sent_new)


