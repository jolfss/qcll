from utils import setup_lm_and_tokenizer
from tokenstring import Tokenstring
import transformers
import random
import string

model, tokenizer = setup_lm_and_tokenizer()
random_str = lambda N : ''.join(random.choice(string.ascii_uppercase + string.digits) for _ in range(N))
T = lambda : Tokenstring(model, tokenizer, transformers.DynamicCache())

def test_operations():
    """Tests the operations on Tokenstrings to ensure consistency between ways of achieving the
    same internal string value."""
    strings = [" ".join(random_str(random.choice([1,2,3,4,5,6])) for _ in range(10)) for _ in range(3)]

    for _string in strings:
        ts_letter_by_letter = T()
        ts_word_by_word = T()
        ts_all_at_once = T()
        ts_overshot_return = T()
        ts_overshot_undershot_return = T()

        for char in _string:
            ts_letter_by_letter.append(char)

        for word in _string.split(" "):
            ts_word_by_word.append(word + " ")
        ts_word_by_word.pop(1)

        ts_all_at_once.pivot(_string)

        ts_overshot_return.pivot(_string).append(_string)
        for _ in _string:
            ts_overshot_return.pop(1)

        ts_overshot_undershot_return.pivot(_string).append(_string)
        for _ in _string:
            ts_overshot_undershot_return.pop(1)
        ts_overshot_undershot_return.pop(10)
        for char in _string[-10:]:
            ts_overshot_undershot_return.append(char)

        # All representations equal
        assert str(ts_letter_by_letter) == str(ts_word_by_word) == str(ts_all_at_once) == str(ts_overshot_return) == str(ts_overshot_undershot_return)  

        # Tokenstring iterables equal 
        for (t1,p1,b1),\
        (t2,p2,b2), \
        (t3,p3,b3), \
        (t4,p4,b4), \
        (t5,p5,b5) in zip(ts_letter_by_letter, 
                        ts_word_by_word, 
                        ts_all_at_once, 
                        ts_overshot_return, 
                        ts_overshot_undershot_return):
            assert t1 == t2 == t3 == t4 == t5
            P = [p1,p2,p3,p4,p5]
            B = [b1,b2,b3,b4,b5]
            for pa,pb in zip(P,P):
                assert abs(pa/pb - 1) <= 1e-8
            for ba,bb in zip(B,B):
                assert abs(ba/bb - 1)<= 1e-8