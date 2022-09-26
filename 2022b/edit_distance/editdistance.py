import edit_distance

def get_edit_distance(string_1, string_2):
    ref = list(string_1)
    hyp = list(string_2)
    sm = edit_distance.SequenceMatcher(a=ref, b=hyp)
    codes = sm.get_opcodes()

    return len(codes)

print(get_edit_distance('leda', 'deal'))

