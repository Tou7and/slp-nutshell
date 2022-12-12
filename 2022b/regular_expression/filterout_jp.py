import re

def example_match_sentence():
    """ 
    The ^ and $ symbols indicate that the matched text must start and end with a Japanese character.
    """
    japanese_regex = re.compile(r"^[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf]+$")
    def is_japanese(word):
        return japanese_regex.match(word) is not None

    japanese_words = ["日本語", "にほんご", "こんにちは"]

    x = "today is a good day. にほんごにほんご"
    jp_words = " ".join(japanese_regex.findall(x))
    print(x)
    print("-"*30)
    print(jp_words)
    print("="*30)

    y = "にほんごにほんご"
    jp_words = " ".join(japanese_regex.findall(y))
    print(y)
    print("-"*30)
    print(jp_words)
    print("="*30)

def example_partial_match():
    japanese_regex = re.compile(r"[\u3000-\u303f\u3040-\u309f\u30a0-\u30ff\uff00-\uff9f\u4e00-\u9faf]+")
    x = "today is a good day. 今天天氣好. にほんご toにほんご"
    jp_words = " ".join(japanese_regex.findall(x))
    print(x)
    print("-"*30)
    print(jp_words)

if __name__ == "__main__":
    example_match_sentence()
