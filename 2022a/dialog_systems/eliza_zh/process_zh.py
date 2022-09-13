import re
import jieba

jieba.load_userdict("dict.txt.big")

with open("doctor_zh.txt", 'r') as reader:
    lines = reader.readlines()

new_lines = []
for line in lines:
    if ":" in line:
        x, y = line.split(":")
        y = y.strip()

        seg_list = jieba.cut(y, cut_all=True)

        y = " ".join(seg_list)

        # y = y.replace()
        y = re.sub(' +', ' ', y)
        new_line = x + ": " + y + "\n"
        new_lines.append(new_line)
        print(new_line)
    else:
        new_lines.append(line)


with open("doctor_zh3.txt", 'w') as writer:
    writer.writelines(new_lines)

