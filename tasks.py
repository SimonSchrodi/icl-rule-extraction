import random
from functools import partial
from collections import Counter
from nltk.metrics import edit_distance

from utils import get_few_shot_prompt

from wordfreq import top_n_list
WORDS = top_n_list('en', 1000)

DIGIT_TO_WORD = {
    "0": "zero",
    "1": "one",
    "2": "two",
    "3": "three",
    "4": "four",
    "5": "five",
    "6": "six",
    "7": "seven",
    "8": "eight",
    "9": "nine"
}

def contains_digit(label: bool, rng: random.Random, low: int= 4, high: int= 8) -> str:
    L = rng.randint(low, high)
    sampled_words = random.sample(WORDS, L)
    for i in range(len(sampled_words)):
        while sampled_words[i].isdigit():
            sampled_words[i] = random.choice(WORDS)
    if label:
        sampled_words[-1] = str(rng.randint(0,9))
    rng.shuffle(sampled_words)
    s = " ".join(sampled_words)
    return s

def contains_digit_counterfactual(word_sequence: str, label: bool, rng: random.Random) -> str:
    words = word_sequence.split()
    if label is True:
        # make sure at least one digit is present
        insert_pos = rng.randint(0, len(words)-1)
        words[insert_pos] = str(rng.randint(0,9))
    else:
        # make sure no digit is present
        for i in range(len(words)):
            while words[i].isdigit():
                words[i] = DIGIT_TO_WORD[words[i]]
    return " ".join(words)

def contains_digit_at_first(label: bool, rng: random.Random, low: int= 4, high: int= 8) -> str:
    L = rng.randint(low, high)
    sampled_words = random.sample(WORDS, L)
    for i in range(1, len(sampled_words)):
        if random.random() < 0.02:
            sampled_words[i] = str(rng.randint(0,9))
    rng.shuffle(sampled_words)
    if label:
        sampled_words[0] = str(rng.randint(0,9))
    s = " ".join(sampled_words)
    return s

def contains_digit_at_first_counterfactual(word_sequence: str, label: bool, rng: random.Random) -> str:
    words = word_sequence.split()
    if label is True:
        words[0] = str(rng.randint(0,9))
    else:
        words[0] = DIGIT_TO_WORD[words[0]]
    return " ".join(words)

def contains_word(label: bool, rng: random.Random, low: int= 4, high: int= 8) -> str:
    L = rng.randint(low, high)
    sampled_words = random.sample(WORDS, L)
    for i in range(len(sampled_words)):
        while "cat" in sampled_words[i]:
            sampled_words[i] = random.choice(WORDS)
    if label:
        sampled_words[-1] = "cat"
    rng.shuffle(sampled_words)
    s = " ".join(sampled_words)
    return s

def contains_word_counterfactual(word_sequence: str, label: bool, rng: random.Random) -> str:
    words = word_sequence.split()
    if label is True:
        insert_pos = rng.randint(0, len(words)-1)
        words[insert_pos] = "cat"
    else:
        for i in range(len(words)):
            while "cat" in words[i]:
                words[i] = random.choice(["feline", "kitty", "tabby", "kitten"])
    return " ".join(words)

def contains_duplicate(label: bool, rng: random.Random, low: int= 4, high: int= 8) -> str:
    L = rng.randint(low, high)
    sampled_words = random.sample(WORDS, L)
    if label:
        sampled_words[-1] = sampled_words[0]
    rng.shuffle(sampled_words)
    s = " ".join(sampled_words)     
    return s

def contains_duplicate_counterfactual(word_sequence: str, label: bool, rng: random.Random) -> str:
    words = word_sequence.split()
    if label is True:
        # make sure at least one duplicate is present
        words[-1] = words[0]
        rng.shuffle(words)
    else:
        # find duplicate words and replace with a word with minimal edit distance
        counter = Counter(words)
        for word, count in counter.items():
            if count > 1:
                min_edit_distance = float('inf')
                replacement = ""
                all_words = WORDS.copy()
                rng.shuffle(all_words)
                for w in all_words:
                    if w != word and edit_distance(word, w) < min_edit_distance:
                        min_edit_distance = edit_distance(word, w)
                        replacement = w
                    if min_edit_distance == 1:
                        break
                words[rng.choice([i for i, w in enumerate(words) if w == word])] = replacement
    return " ".join(words)

def contains_only_words_with_same_starting_letter(label: bool, rng: random.Random, low: int= 4, high: int= 8) -> str:
    L = rng.randint(low, high)
    sampled_words = random.sample(WORDS, L)
    if label:
        first_letter = sampled_words[0][0]
        for i in range(1, L):
            while not sampled_words[i].startswith(first_letter):
                sampled_words[i] = random.choice(WORDS)
    else:
        first_letter = sampled_words[0][0]
        for i in range(1, L):
            while sampled_words[i].startswith(first_letter):
                sampled_words[i] = random.choice(WORDS)
    rng.shuffle(sampled_words)
    s = " ".join(sampled_words)     
    return s

def contains_only_words_with_same_starting_letter_counterfactual(word_sequence: str, label: bool, rng: random.Random) -> str:
    words = word_sequence.split()
    if label is True:
        first_letter = words[0][0]
        for i in range(1, len(words)):
            while not words[i].startswith(first_letter):
                words[i] = random.choice(WORDS)
        rng.shuffle(words)
    else:
        first_letter = words[0][0]
        rand_index = rng.randint(0, len(words)-1)
        while words[rand_index].startswith(first_letter):
            words[rand_index] = random.choice(WORDS)
    return " ".join(words)

def sorted_words_starting_letters(label: bool, rng: random.Random, low: int= 4, high: int= 8) -> list[str]:
    L = rng.randint(low, high)
    sampled_words = random.sample(WORDS, L)
    if label:
        sampled_words = sorted(sampled_words, key=lambda w: w[0])
    else:
        while sorted(sampled_words, key=lambda w: w[0]) == sampled_words:
            random.shuffle(sampled_words)
    return " ".join(sampled_words)

def sorted_words_starting_letters_counterfactual(word_sequence: str, label: bool, rng: random.Random) -> str:
    words = word_sequence.split()
    if label is True:
        words = sorted(words, key=lambda w: w[0])
    else:
        rand_index = rng.randint(1, len(words)-1)
        words[rand_index], words[rand_index-1] = words[rand_index-1], words[rand_index]
    return " ".join(words)

def is_palindrome(label: bool, rng: random.Random, low: int= 4, high: int= 8) -> str:
    L = rng.randint(low, high)
    sampled_words = random.choices("abcdefghijklmnopqrstuvwxyz", k=(L+1)//2)
    if label:
        palin_words = sampled_words + sampled_words[::-1]
    else:
        palin_words = sampled_words + random.choices("abcdefghijklmnopqrstuvwxyz", k=(L+1)//2)
        while palin_words == palin_words[::-1]:
            palin_words[-1] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return " ".join(palin_words)

def is_palindrome_counterfactual(word_sequence: str, label: bool, rng: random.Random) -> str:
    words = word_sequence.split()
    if label is True:
        half = len(words)//2
        words = words[:half] + words[:half][::-1]
    else:
        rand_index = rng.randint(0, len(words)-1)
        prev_char = words[rand_index]
        while words[rand_index] == prev_char:
            words[rand_index] = random.choice("abcdefghijklmnopqrstuvwxyz")
    return " ".join(words)

def is_even_length(label: bool, rng: random.Random, low: int= 4, high: int= 8) -> str:
    L = rng.randint(low, high)
    if label:
        if L % 2 != 0:
            L += 1
    else:
        if L % 2 == 0:
            L += 1
    sampled_words = random.sample(WORDS, L)
    s = " ".join(sampled_words)     
    return s

def is_even_length_counterfactual(word_sequence: str, label: bool, rng: random.Random) -> str:
    words = word_sequence.split()
    if label is True:
        if len(words) % 2 != 0:
            words.append(random.choice(WORDS))
    else:
        if len(words) % 2 == 0:
            words.append(random.choice(WORDS))
    return " ".join(words)

def all_words_start_with_vowel(label: bool, rng: random.Random, low: int= 4, high: int= 8) -> str:
    L = rng.randint(low, high)
    sampled_words = random.sample(WORDS, L)
    vowels = set('aeiou')
    if label:
        for i in range(L):
            while sampled_words[i][0].lower() not in vowels:
                sampled_words[i] = random.choice(WORDS)
    else:
        while all(sampled_words[i][0].lower() in vowels for i in range(L)):
            sampled_words = random.sample(WORDS, L)
    rng.shuffle(sampled_words)
    s = " ".join(sampled_words)     
    return s

def all_words_start_with_vowel_counterfactual(word_sequence: str, label: bool, rng: random.Random) -> str:
    words = word_sequence.split()
    vowels = set('aeiou')
    if label is True:
        for i in range(len(words)):
            while words[i][0].lower() not in vowels:
                words[i] = random.choice(WORDS)
    else:
        rand_index = rng.randint(0, len(words)-1)
        while words[rand_index][0].lower() in vowels:
            words[rand_index] = random.choice(WORDS)
    return " ".join(words)

def is_tab_separator(label: bool, rng: random.Random, low: int= 4, high: int= 8) -> str:
    L = rng.randint(low, high)
    sampled_words = random.sample(WORDS, L)
    if label:
        s = "\t".join(sampled_words)     
    else:
        s = " ".join(sampled_words)     
    return s

def is_tab_separator_counterfactual(word_sequence: str, label: bool, rng: random.Random) -> str:
    if label is True:
        words = word_sequence.split()
        return "\t".join(words)
    else:
        words = word_sequence.split("\t")
        return " ".join(words)

RULES = {
    "contains_digit": (contains_digit, "Label True iff the input contains at least one digit (0-9).", contains_digit_counterfactual),
    "contains_digit_at_first": (contains_digit_at_first, "Label True iff the input contains a digit (0-9) as the first word.", contains_digit_at_first_counterfactual),
    "contains_word": (contains_word, "Label True iff the input contains the word 'cat'.", contains_word_counterfactual),
    "contains_duplicate": (contains_duplicate, "Label True iff the input contains at least one duplicated word.", contains_duplicate_counterfactual),
    "contains_only_words_with_same_starting_letter": (contains_only_words_with_same_starting_letter, "Label True iff the input contains only words that start with the same letter.", contains_only_words_with_same_starting_letter_counterfactual),
    "sorted_words_starting_letters": (sorted_words_starting_letters, "Label True iff the words in the input are sorted in ascending order based on their starting letters.", sorted_words_starting_letters_counterfactual),
    "is_palindrome": (is_palindrome, "Label True iff the input is a palindrome (reads the same forwards and backwards).", is_palindrome_counterfactual),
    "is_even_length": (is_even_length, "Label True iff the input contains an even number of words.", is_even_length_counterfactual),
    "all_words_start_with_vowel": (all_words_start_with_vowel, "Label True iff all words in the input start with a vowel (a, e, i, o, u).", all_words_start_with_vowel_counterfactual),
    "is_tab_separator": (is_tab_separator, "Label True iff the words in the input are separated by tab (`\\t`).", is_tab_separator_counterfactual),
}

RULES_ALTERNATIVES_CHOICES = {
    "contains_digit": [
        "Label True iff the input contains no digits.",
        "Label True iff the input contains only digits.",
        "Label True iff the input contains at least one vowel.",
    ],
    "contains_digit_at_first": [
        "Label True iff the input contains a digit (0-9) as the last word.",
        "Label True iff the input contains no digits.",
        "Label True iff the input contains at least one digit (0-9).",
    ],
    "contains_word": [
        "Label True iff the input contains no instances of the word 'cat'.",
        "Label True iff the input contains the word 'dog'.",
        "Label True iff the input contains the word 'kitten'.",
    ],
    "contains_duplicate": [
        "Label True iff the input contains all unique words.",
        "Label True iff the input contains only words starting with the same letter.",
        "Label True iff the input contains only the same words.",
    ],
    "contains_only_words_with_same_starting_letter": [
        "Label True iff the input contains words that start with different letters.",
        "Label True iff the input contains only words that end with the same letter.",
        "Label True iff the input contains only the same words.",
    ],
    "sorted_words_starting_letters": [
        "Label True iff the words in the input are sorted in descending order based on their starting letters.",
        "Label True iff the words in the input are sorted in ascending order based on their lengths.",
        "Label True iff the words in the input are sorted in ascending order based on their last letters.",
    ],
    "is_palindrome": [
        "Label True iff the input starts and ends with the same letter.",
        "Label True iff the input contains each letter at least twice.",
        "Label True iff the input is of equal length.",
    ],
    "is_even_length": [
        "Label True iff the input contains an odd number of words.",
        "Label True iff the input contains a prime number of words.",
        "Label True iff the input contains more than five words.",
    ],
    "all_words_start_with_vowel": [
        "Label True iff all words in the input start with a consonant.",
        "Label True iff at least one word in the input starts with a vowel (a, e, i, o, u).",
        "Label True iff all words in the input end with a vowel (a, e, i, o, u).",
    ],
    "is_tab_separator": [
        "Label True iff the words in the input are separated by spaces.",
        "Label True iff the words in the input are separated by commas.",
        "Label True iff the words in the input are separated by semicolons.",
    ],
}

def synthesize(rule_name:str, n:int, seed:int=0)->list:
    rng = random.Random(seed)
    gen, rule, gen_cf = RULES[rule_name]
    out=[]
    for i in range(n):
        y = True if i%2==0 else False
        s = gen(y, rng)
        out.append((s, y))
    return out

def fewshot(rule_name:str, k:int, seed:int=42)->str:
    data = synthesize(rule_name, k, seed=seed)
    data = [(d[0], str(d[1])) for d in data]
    rng = random.Random(seed)
    rng.shuffle(data)
    return get_few_shot_prompt(data)

def sample_test(rule_name:str, m:int, seed:int=43)->list[str]:
    return synthesize(rule_name, m, seed=seed)

def label(rule_name:str, s:str)->bool:
    return RULES[rule_name][0](s)
