"""
Install:
    pip install edit-distance

Reference:
    https://pypi.org/project/edit-distance/
"""
import edit_distance

def get_edit_distance_w(sent1, sent2):
    ref = sent1.split(" ")
    hyp = sent2.split(" ")
    sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
    codes = sm.get_opcodes()

    print(codes)
    return len(codes)

print(get_edit_distance('leda', 'deal'))
print(get_edit_distance('leda', 'deala'))
print(get_edit_distance_w('leda', 'deal'))

