spanish_phones_grouped = {
    "plosives": ["p", "t", "k", "b", "d", "g", "G", "cl"],  # G = voiced velar?
    "nasals": ["m", "n", "N"],  # N = [ŋ] (velar nasal)
    "fricatives": ["s", "f", "x", "z", "h", "D"],  # 0 = θ (theta), 3 = [ʒ], D = ð
    "affricates": ["ch", "jh"],  # ch = [tʃ], jh = [dʒ]
    "approximants": ["B", "Y", "w"],  # B = Spanish [β̞], Y = [ʝ]?, w = glide
    "laterals": ["l", "ll"],  # ll = palatal lateral [ʎ]?
    "rhotics": ["r", "rr"],  # r = tap [ɾ], rr = trill [r]
    "palatal/frontal glides": ["y"],  # may be glide [j]
    "aspirates": ["h"],  # possibly redundant with fricatives
    "breathing": ["AP", "sh"],  # AP? Possibly pause or closure? sh = [ʃ] (Catalan/loanword)
    "vowels": ['a', 'e', 'i', 'o', 'u']
}

spanish_consonants_by_place = {
    "bilabial": ["p", "b", "m", "B", "w"],       # B = [β̞]
    "labiodental": ["f"],
    "dental": ["t", "d", "D"],              # D = [ð], 0 = [θ] wrong?
    "alveolar": ["s", "z", "n", "l", "r", "rr"], # r = [ɾ], rr = [r]
    "postalveolar": ["sh", "ch", "jh"],     # sh = [ʃ], ch = [tʃ], jh = [dʒ], 3 = [ʒ] wrong
    "palatal": ["ll", "y", "Y"],                 # ll = [ʎ], y/Y = [ʝ]/[j]
    "velar": ["k", "g", "G", "x", "N"],           # G = [ɣ], x = [χ]/[x], N = [ŋ]
    "glottal": ["h"],
    "unknown/other": ["AP", "cl"]                # AP = pause?, cl = closure?
}

arpabet_consonants = {
    "stops": ["P", "B", "T", "D", "K", "G"],
    "affricates": ["CH", "JH"],
    "fricatives1": ["F", "V", "TH", "DH"],
    "fricatives2": ["S", "Z", "SH", "ZH"],
    "nasals": ["M", "N", "NG"],
    "liquids": ["L", "R"],
    "glides": ["W", "Y"],
    "aspirate": ["HH"],
    "silence": ['sp']
}

arpabet_consonants = dict(sorted(arpabet_consonants.items()))

arpabet_vowels = {
    "high_front": ["IY", "IH"],
    "mid_front": ["EY", "EH"],
    "low_front": ["AE"],
    "mid_central": ["AH", "ER"],
    "high_back": ["UW", "UH"],
    "mid_back": ["OW", "AO"],
    "low_back": ["AA"],
    "diphthongs": ["AY", "AW", "OY"],
    "silence": ['sp']
}

spanglish_vowels = { #"AH"
    "a": ["AE", "AA"],
    "e": ["EY", "EH", "ER"],
    "i": ["IY", "IH"],
    "o": ["OW", "AO"],
    "u": ["UW", "UH"],
    "diphthongs": ["AY", "AW", "OY"],
    "silence": ['sp']
}

english_consonants_by_place = {
    "bilabial": ["P", "B", "M"],
    "labiodental": ["F", "V"],
    "dental": ["TH", "DH"],
    "alveolar": ["T", "D", "S", "Z", "N", "L", "R"],
    "postalveolar": ["SH", "ZH", "CH", "JH"],
    "palatal": ["Y"],
    "velar": ["K", "G", "NG"],
    "glottal": ["HH"],
    "labiovelar": ["W"]
}

arpabet_vowels = dict(sorted(arpabet_vowels.items()))
arpabet = dict(sorted(arpabet_vowels.items())) | dict(sorted(arpabet_consonants.items()))

spanish_groups_PM = {
"Labial": ['f', 'p', 'b', 'm'],
"Dental": ['z', 't', 'd'],
"Alveolar": ['s', 'n', 'l', 'r', 'rr'],
"Palatal": ['ch', 'y', 'll', 'i', 'e'], #'ñ'
"Velar": ['j', 'k', 'g', 'u', 'o'],
'a': ['a']
#'resp': ['SP']
}

spanish_groups_vowels_consonants = {
"vowels": ['a', 'e','i', 'o', 'u'],
"consonants": ['f', 'p', 'b', 'm', 'z', 't', 'd',
               's', 'n', 'l', 'r', 'rr',
               'ch', 'y', 'll',
               'j', 'k', 'g'],
"breathing": ['SP', 'AP']
}

spanish_groups_vowels_consonants_v2 = {
"a": ['a'],
"e": ['e'],
"i": ['i'],
"o": ['o'],
"u": ['u'],
"consonants": ['f', 'p', 'b', 'm', 'z', 't', 'd',
               's', 'n', 'l', 'r', 'rr',
               'ch', 'y', 'll',
               'j', 'k', 'g'],
"breathing": ['SP', 'AP']
}