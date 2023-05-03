from enum import Enum
import re


class Symbol(Enum):
    UNKNOWN = '< UNK >'
    BEGIN   = '< beg >'
    END     = '< end >'


class PseudoWords(Enum):
    TWO_DIGIT_NUM               = '< TWO_DIGIT_NUM >'
    FOUR_DIGIT_NUM              = '< FOUR_DIGIT_NUM >'
    OTHER_NUM                   = '< NUMBER >'

    CONTAINS_DIGIT_AND_ALPHA    = '< CONTAINS_DIGIT_AND_ALPHA >'
    CONTAINS_DIGIT_AND_DASH     = '< CONTAINS_DIGIT_AND_DASH >'
    CONTAINS_DIGIT_AND_SLASH    = '< CONTAINS_DIGIT_AND_SLASH >'
    CONTAINS_DIGIT_AND_COMMA    = '< CONTAINS_DIGIT_AND_COMMA >'
    CONTAINS_DIGIT_AND_PERIOD   = '< CONTAINS_DIGIT_AND_PERIOD >'
    
    ALL_CAPS                    = '< ALL_CAPS >'
    CAP_PERIOD                  = '< CAP_PERIOD >'
    CAP_INIT_ENDING_ED         = '< CAP_INIT_ENDING_ED >'
    CAP_INIT_ENDING_LY         = '< CAP_INIT_ENDING_LY >'
    CAP_INIT_ENDING_IVE        = '< CAP_INIT_ENDING_IVE >'
    CAP_INIT_ENDING_ION        = '< CAP_INIT_ENDING_ION >'
    CAP_INIT_ENDING_S          = '< CAP_INIT_ENDING_S >'
    CAP_INIT_ENDING_IAL        = '< CAP_INIT_ENDING_IAL >'
    CAP_INIT_ENDING_TY         = '< CAP_INIT_ENDING_TY >'
    CAP_INIT_ENDING_ING        = '< CAP_INIT_ENDING_ING >'
    CAP_INIT_ENDING_ER         = '< CAP_INIT_ENDING_ER >'
    CAP_INIT                    = '< CAP_INIT >'

    LOWERCASE_ENDING_ED         = '< LOWERCASE_ENDING_ED >'
    LOWERCASE_ENDING_LY         = '< LOWERCASE_ENDING_LY >'
    LOWERCASE_ENDING_IVE        = '< LOWERCASE_ENDING_IVE >'
    LOWERCASE_ENDING_ION        = '< LOWERCASE_ENDING_ION >'
    LOWERCASE_ENDING_NESS       = '< LOWERCASE_ENDING_NESS >'
    LOWERCASE_ENDING_OUS        = '< LOWERCASE_ENDING_OUS >'
    LOWERCASE_ENDING_INGS       = '< LOWERCASE_ENDING_INGS >'
    LOWERCASE_ENDING_S          = '< LOWERCASE_ENDING_S >'
    LOWERCASE_ENDING_IAL        = '< LOWERCASE_ENDING_IAL >'
    LOWERCASE_ENDING_TY         = '< LOWERCASE_ENDING_TY >'
    LOWERCASE_ENDING_ING        = '< LOWERCASE_ENDING_ING >'
    LOWERCASE_ENDING_ER         = '< LOWERCASE_ENDING_ER >'
    LOWERCASE                   = '< LOWERCASE >'

    CONTAINS_DIGIT_AND_COLON    = '< CONTAINS_DIGIT_AND_COLON >'

    # NUM_DASH_ALPHA              = '< NUM_DASH_ALPHA >'
    ALPHA_DASH_ALPHA_ED         = '< ALPHA_DASH_ALPHA_ED >'
    ALPHA_DASH_ALPHA_LY         = '< ALPHA_DASH_ALPHA_LY >'
    ALPHA_DASH_ALPHA_IVE        = '< ALPHA_DASH_ALPHA_IVE >'
    ALPHA_DASH_ALPHA_ION        = '< ALPHA_DASH_ALPHA_ION >'
    ALPHA_DASH_ALPHA_S          = '< ALPHA_DASH_ALPHA_S >'
    ALPHA_DASH_ALPHA_IAL        = '< ALPHA_DASH_ALPHA_IAL >'
    ALPHA_DASH_ALPHA_TY         = '< ALPHA_DASH_ALPHA_TY >'
    ALPHA_DASH_ALPHA_ING        = '< ALPHA_DASH_ALPHA_ING >'
    ALPHA_DASH_ALPHA_ER         = '< ALPHA_DASH_ALPHA_ER >'
    CONTAINS_ALPHA_AND_DASH     = '< CONTAINS_ALPHA_AND_DASH >'
    ACRONYM                     = '< ACRONYM >'
    # FIRST_WORD                  = '< FIRST_WORD >'


def match(word: str) -> str:
    RE_TWO_DIGIT_NUM               = re.compile(r'^[0-9]{2}$')
    RE_FOUR_DIGIT_NUM              = re.compile(r'^[0-9]{4}$')
    RE_NUMBER                       = re.compile(r'^[0-9]+$')
    RE_CONTAINS_DIGIT_AND_ALPHA    = re.compile(r'^(?=.*[0-9])(?=.*[a-zA-Z])([a-zA-Z0-9]+)$')
    RE_CONTAINS_DIGIT_AND_DASH     = re.compile(r'^[0-9-]+$')
    RE_CONTAINS_DIGIT_AND_SLASH    = re.compile(r'^[0-9/]+$')
    RE_CONTAINS_DIGIT_AND_COMMA    = re.compile(r'^[0-9,]+$')
    RE_CONTAINS_DIGIT_AND_PERIOD   = re.compile(r'^[0-9\.]+$')
    RE_ALL_CAPS                    = re.compile(r'^[A-Z]+$')
    RE_CAP_PERIOD                  = re.compile(r'^[A-Z]\.$')
    
    RE_CAP_INIT_ENDING_ED          = re.compile(r'[A-Z][a-z]+ed')
    RE_CAP_INIT_ENDING_LY          = re.compile(r'[A-Z][a-z]+ly')
    RE_CAP_INIT_ENDING_IVE         = re.compile(r'[A-Z][a-z]+ive')
    RE_CAP_INIT_ENDING_ION         = re.compile(r'[A-Z][a-z]+ion')
    RE_CAP_INIT_ENDING_S           = re.compile(r'[A-Z][a-z]+s')
    RE_CAP_INIT_ENDING_IAL         = re.compile(r'[A-Z][a-z]+ial')
    RE_CAP_INIT_ENDING_TY          = re.compile(r'[A-Z][a-z]+ty')
    RE_CAP_INIT_ENDING_ING         = re.compile(r'[A-Z][a-z]+ing')
    RE_CAP_INIT_ENDING_ER          = re.compile(r'[A-Z][a-z]+er')
    RE_CAP_INIT                    = re.compile(r'^[A-Z][a-z]*$')

    RE_LOWERCASE_ENDING_ED         = re.compile(r'^[a-z]+ed$')
    RE_LOWERCASE_ENDING_LY         = re.compile(r'^[a-z]+ly$')
    RE_LOWERCASE_ENDING_IVE        = re.compile(r'^[a-z]+ive$')
    RE_LOWERCASE_ENDING_ION        = re.compile(r'^[a-z]+ion$')
    RE_LOWERCASE_ENDING_NESS       = re.compile(r'^[a-z]+ness$')
    RE_LOWERCASE_ENDING_OUS        = re.compile(r'^[a-z]+ous$')
    RE_LOWERCASE_ENDING_INGS       = re.compile(r'^[a-z]+ings$')
    RE_LOWERCASE_ENDING_S          = re.compile(r'^[a-z]+s$')
    RE_LOWERCASE_ENDING_IAL        = re.compile(r'^[a-z]+ial$')
    RE_LOWERCASE_ENDING_TY         = re.compile(r'^[a-z]+ty$')
    RE_LOWERCASE_ENDING_ING        = re.compile(r'^[a-z]+ing$')
    RE_LOWERCASE_ENDING_ER         = re.compile(r'^[a-z]+er$')
    RE_LOWERCASE                   = re.compile(r'^[a-z]+$')

    RE_CONTAINS_DIGIT_AND_COLON    = re.compile(r'^[0-9:]+$')

    RE_ALPHA_DASH_ALPHA_ED         = re.compile(r'^[a-z]+-[a-z]+ed$')
    RE_ALPHA_DASH_ALPHA_LY         = re.compile(r'^[a-z]+-[a-z]+ly$')
    RE_ALPHA_DASH_ALPHA_IVE        = re.compile(r'^[a-z]+-[a-z]+ive$')
    RE_ALPHA_DASH_ALPHA_ION        = re.compile(r'^[a-z]+-[a-z]+ion$')
    RE_ALPHA_DASH_ALPHA_S          = re.compile(r'^[a-z]+-[a-z]+s$')
    RE_ALPHA_DASH_ALPHA_IAL        = re.compile(r'^[a-z]+-[a-z]+ial$')
    RE_ALPHA_DASH_ALPHA_TY         = re.compile(r'^[a-z]+-[a-z]+ty$')
    RE_ALPHA_DASH_ALPHA_ING        = re.compile(r'^[a-z]+-[a-z]+ing$')
    RE_ALPHA_DASH_ALPHA_ER         = re.compile(r'^[a-z]+-[a-z]+er$')
    RE_CONTAINS_ALPHA_AND_DASH     = re.compile(r'^([a-zA-Z0-9\.]*-[a-zA-Z0-9\.]*)+$')
    RE_ACRONYM                     = re.compile(r'(?:[a-zA-Z]\.){2,}')

    match_sequence = [
        # Number Series
        RE_TWO_DIGIT_NUM            ,
        RE_FOUR_DIGIT_NUM           ,
        RE_NUMBER                   ,
        RE_CONTAINS_DIGIT_AND_ALPHA ,
        RE_CONTAINS_DIGIT_AND_DASH  ,
        RE_CONTAINS_DIGIT_AND_SLASH ,
        RE_CONTAINS_DIGIT_AND_COMMA ,
        RE_CONTAINS_DIGIT_AND_PERIOD,

        # Uppercase Series
        RE_ALL_CAPS                 ,
        RE_CAP_PERIOD               ,
        RE_CAP_INIT_ENDING_ED       ,
        RE_CAP_INIT_ENDING_LY       ,
        RE_CAP_INIT_ENDING_IVE      ,
        RE_CAP_INIT_ENDING_ION      ,
        RE_CAP_INIT_ENDING_S        ,
        RE_CAP_INIT_ENDING_IAL      ,
        RE_CAP_INIT_ENDING_TY       ,
        RE_CAP_INIT_ENDING_ING      ,
        RE_CAP_INIT_ENDING_ER       ,
        RE_CAP_INIT                 ,

        # Lowercase Series
        RE_LOWERCASE_ENDING_ED      ,
        RE_LOWERCASE_ENDING_LY      ,
        RE_LOWERCASE_ENDING_IVE     ,
        RE_LOWERCASE_ENDING_ION     ,
        RE_LOWERCASE_ENDING_NESS    ,
        RE_LOWERCASE_ENDING_OUS     ,
        RE_LOWERCASE_ENDING_INGS    ,
        RE_LOWERCASE_ENDING_S       ,
        RE_LOWERCASE_ENDING_IAL     ,
        RE_LOWERCASE_ENDING_TY      ,
        RE_LOWERCASE_ENDING_ING     ,
        RE_LOWERCASE_ENDING_ER      ,
        RE_LOWERCASE                ,
        RE_CONTAINS_DIGIT_AND_COLON ,

        # Alpha Dash Series
        RE_ALPHA_DASH_ALPHA_ED      ,
        RE_ALPHA_DASH_ALPHA_LY      ,
        RE_ALPHA_DASH_ALPHA_IVE     ,
        RE_ALPHA_DASH_ALPHA_ION     ,
        RE_ALPHA_DASH_ALPHA_S       ,
        RE_ALPHA_DASH_ALPHA_IAL     ,
        RE_ALPHA_DASH_ALPHA_TY      ,
        RE_ALPHA_DASH_ALPHA_ING     ,
        RE_ALPHA_DASH_ALPHA_ER      ,
        RE_CONTAINS_ALPHA_AND_DASH  ,
        RE_ACRONYM
    ]

    for type_, expression in zip(PseudoWords, match_sequence):
        if expression.match(word):
            return type_.value
    return Symbol.UNKNOWN.value


class PseudoWordUnknownHandler:

    def transform_freq(self, frequency: dict[str, int], threshold: int) -> dict[str, int]:
        data = { key: val for key, val in frequency.items() if val >= threshold }
        leftovers = { key: val for key, val in frequency.items() if val < threshold }

        pseudoword_dict = {pseudoword.value: 0 for pseudoword in PseudoWords} | {Symbol.UNKNOWN.value: 0}

        for word, count in leftovers.items():
            pseudoword_dict[match(word)] += count
        
        return data | pseudoword_dict

    def replace(self, sequences: list[list[str]], state_space: set[str]) -> list[list[str]]:
        result: list[list[str]] = []
        # Replace unknown words with appropriate symbol
        for i, sentence in enumerate(sequences):
            result.append([])
            for word in sentence:
                if word not in state_space:
                    result[i].append(match(word))
                else:
                    result[i].append(word)
        
        return result
    