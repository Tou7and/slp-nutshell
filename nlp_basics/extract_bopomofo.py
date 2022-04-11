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

def extract_not_mandarin(text):
    pattern = re.compile(u'[^⺀-⺙⺛-⻳⼀-⿕々〇〡-〩〸-〺〻㐀-䶵一-鿃豈-鶴侮-頻並-龎]')
    results = pattern.finditer(text)
    zh_chars = []
    for result in results:
        print(result.group(), result.span())
        zh_chars.append(result.group())
    sent_new = "".join(zh_chars)
    print()
    print(sent_new)
    return sent_new

def extract_bopomofo(text):
    pattern = re.compile(u'[ㄅㄈㄋㄏㄊㄑㄙㄚㄔㄆㄣ𠄎ㄘㄌㄨㄠㄛㄥㄡㄗㄧㄇㄟㄢㄩㄍㄎㄒㄞㄜㄉㄓㄖㄐ𠃋ㄕㄦㄝㄤ𡿨]')
    results = pattern.finditer(text)

    zh_chars = []
    for result in results:
        # print(result.group(), result.span())
        zh_chars.append(result.group())
    sent_new = "".join(zh_chars)
    # print()
    # print(sent_new)
    return sent_new


x = """
ㄅ	From 勹, the ancient form and current top portion of 包 bāo, "to wrap up; package"	p	b	p	包 bāo
ㄅㄠ
ㄆ	From 攵, a variant form of 攴 pū, "to knock lightly".	pʰ	p	pʻ	撲 pū
ㄆㄨ
ㄇ	From 冂, the archaic character and current "cover" radical 冖 mì.	m	m	m	冞 mí
ㄇㄧˊ
ㄈ	From "right open box" radical 匚 fāng.	f	f	f	匪 fěi
ㄈㄟˇ
ㄉ	From 𠚣, archaic form of 刀 dāo. Compare the Shuowen seal 刀-seal.svg.	t	d	t	地 dì
ㄉㄧˋ
ㄊ	From ㄊ tū, an upside-down form of 子 zǐ and an ancient form of 突 tū (Shuowen Seal Radical 528.svg and Shuowen Seal Radical 525.svg in seal script)[8][9]	tʰ	t	tʻ	提 tí
ㄊㄧˊ
ㄋ	From 乃-seal.svg/𠄎, ancient form of 乃 nǎi (be)	n	n	n	你 nǐ
ㄋㄧˇ
ㄌ	From 𠠲, archaic form of 力 lì	l	l	l	利 lì
ㄌㄧˋ
ㄍ	From the obsolete character 巜 guì/kuài "river"	k	g	k	告 gào
ㄍㄠˋ
ㄎ	From the archaic character, now "breath" or "sigh" component 丂 kǎo	kʰ	k	kʻ	考 kǎo
ㄎㄠˇ
ㄏ	From the archaic character and current radical 厂 hǎn	x	h	h	好 hǎo
ㄏㄠˇ
ㄐ	From the archaic character 丩 jiū	tɕ	j	ch	叫 jiào
ㄐㄧㄠˋ
ㄑ	From the archaic character 𡿨 quǎn, graphic root of the character 巛 chuān (modern 川)	tɕʰ	q	chʻ	巧 qiǎo
ㄑㄧㄠˇ
ㄒ	From 丅, an ancient form of 下 xià.	ɕ	x	hs	小 xiǎo
ㄒㄧㄠˇ
ㄓ	From 之-seal.svg/𡳿, archaic form of 之 zhī.	ʈʂ	zhi, zh-	ch	知 zhī
ㄓ;
主 zhǔ
ㄓㄨˇ
ㄔ	From the character and radical 彳 chì	ʈʂʰ	chi, ch-	chʻ	吃 chī
ㄔ;
出 chū
ㄔㄨ
ㄕ	From 𡰣, an ancient form of 尸 shī	ʂ	shi, sh-	sh	是 shì
ㄕˋ;
束 shù
ㄕㄨˋ
ㄖ	Modified from the seal script 日-seal.svg form of 日 rì (day/sun)	ɻ~ʐ	ri, r-	j	日 rì
ㄖˋ;
入 rù
ㄖㄨˋ
ㄗ	From the archaic character and current radical 卩 jié, dialectically zié ([tsjě]; tsieh² in Wade–Giles)	ts	zi, z-	ts	字 zì
ㄗˋ;
在 zài
ㄗㄞˋ
ㄘ	From 𠀁, archaic form of 七 qī, dialectically ciī ([tsʰí]; tsʻi¹ in Wade–Giles). Compare semi-cursive form Qi1 seven semicursive.png and seal-script 七-seal.svg.	tsʰ	ci, c-	tsʻ	詞 cí
ㄘˊ;
才 cái
ㄘㄞˊ
ㄙ	From the archaic character 厶 sī, which was later replaced by its compound 私 sī.	s	si, s-	s	四 sì
ㄙˋ;
塞 sāi
ㄙㄞ
Rhymes and medials
Bopomofo	Origin	IPA	Pinyin	WG	Example
ㄚ	From 丫 yā	a	a	a	大 dà
ㄉㄚˋ
ㄛ	From the obsolete character 𠀀 hē, inhalation, the reverse of 丂 kǎo, which is preserved as a phonetic in the compound 可 kě.[10]	o	o	o	多 duō
ㄉㄨㄛ
ㄜ	Derived from its allophone in Standard Chinese, ㄛ o	ɤ	e	o/ê	得 dé
ㄉㄜˊ
ㄝ	From 也 yě (also). Compare the Warring States bamboo form Ye3 also chu3jian3 warring state of chu3 small.png	e	-ie/ê	eh	爹 diē
ㄉㄧㄝ
ㄞ	From 𠀅 hài, archaic form of 亥.	ai	ai	ai	晒 shài
ㄕㄞˋ
ㄟ	From 乁 yí, an obsolete character meaning 移 yí "to move".	ei	ei	ei	誰 shéi
ㄕㄟˊ
ㄠ	From 幺 yāo	au	ao	ao	少 shǎo
ㄕㄠˇ
ㄡ	From 又 yòu	ou	ou	ou	收 shōu
ㄕㄡ
ㄢ	From the archaic character 𢎘 hàn "to bloom", preserved as a phonetic in the compound 犯 fàn	an	an	an	山 shān
ㄕㄢ
ㄣ	From 𠃉, archaic variant of 鳦 yǐ or 乚 yà[11] (乚 is yǐn according to other sources[12])	ən	en	ên	申 shēn
ㄕㄣ
ㄤ	From 尢 wāng	aŋ	ang	ang	上 shàng
ㄕㄤˋ
ㄥ	From 𠃋, archaic form of 肱 gōng[13]	əŋ	eng	êng	生 shēng
ㄕㄥ
ㄦ	From 儿, the bottom portion of 兒 ér used as a cursive and simplified form	aɚ	er	êrh	而 ér
ㄦˊ
ㄧ	From 一 yī (one)	i	yi, -i	i	以 yǐ
ㄧˇ;
逆 nì
ㄋㄧˋ
ㄨ	From 㐅, ancient form of 五 wǔ (five). Compare the transitory form 𠄡.	u	w, wu, -u	u/w	努 nǔ
ㄋㄨˇ;
我 wǒ
ㄨㄛˇ
ㄩ	From the ancient character 凵 qū, which remains as a radical	y	yu, -ü	ü/yü	雨 yǔ
ㄩˇ;
女 nǚ
ㄋㄩˇ
ㄭ
MoeKai Bopomofo U+312D.svg	From the character 帀. It represents the minimal vowel of ㄓ，ㄔ，ㄕ，ㄖ，ㄗ，ㄘ，ㄙ， though it is not used after them in transcription.[14]	ɻ̩~ʐ̩, ɹ̩~z̩	-i	ih/ŭ	資 zī
ㄗ;
知 zhī
ㄓ;
死 sǐ
ㄙ
"""

y = extract_bopomofo(x)
print(y)

print()
sent = "Steve Jobs手上拿著一隻iphone 好潮ㄛ 褲子都濕ㄌ"
print(extract_bopomofo(sent))
