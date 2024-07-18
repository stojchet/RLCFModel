import random
import re

from datasets import Dataset

java_keywords_misspellings = {
    "abstract": ["abstrcat", "abstact", "abstrac"],
    "assert": ["assret", "asert", "assertt"],
    "boolean": ["boelan", "bollean", "booean"],
    "break": ["brek", "brak", "breeak"],
    "byte": ["byet", "btye", "bytte"],
    "case": ["csae", "csaes", "caes"],
    "catch": ["ctach", "cathc", "catc"],
    "char": ["cahr", "chaar", "chr"],
    "class": ["clss", "cllas", "clsss"],
    "const": ["conts", "cost", "constt"],
    "continue": ["conitnue", "cntinue", "contnue"],
    "default": ["defualt", "deafult", "dfault"],
    "do": ["d", "od", "doe"],
    "double": ["doubl", "duble", "doubel"],
    "else": ["els", "elese", "elsee"],
    "enum": ["enmu", "enum", "emum"],
    "extends": ["exetnds", "extnds", "extens"],
    "final": ["fianl", "fnal", "finala"],
    "finally": ["fianlly", "finaly", "finalyl"],
    "float": ["flaot", "flot", "floaat"],
    "for": ["fro", "forr", "foor"],
    "goto": ["gto", "goot", "gotoo"],
    "if": ["f", "iff", "fi"],
    "implements": ["imlpements", "implments", "implemnts"],
    "import": ["improt", "impor", "importt"],
    "instanceof": ["instnaceof", "instaceof", "instancef"],
    "int": ["itn", "it", "innt"],
    "interface": ["interfcae", "inetrface", "interace"],
    "long": ["lon", "lng", "loong"],
    "native": ["natvie", "natve", "natie"],
    "new": ["ne", "nwe", "neew"],
    "package": ["pakcage", "pacakge", "packge"],
    "private": ["privte", "privat", "pirvate"],
    "protected": ["proetcted", "proected", "prtected"],
    "public": ["publc", "pubic", "pblic"],
    "return": ["retun", "retur", "rturn"],
    "short": ["shrot", "shot", "sort"],
    "static": ["sttaic", "sttic", "staic"],
    "strictfp": ["strictp", "stricfp", "strictf"],
    "super": ["supe", "supr", "supper"],
    "switch": ["swtich", "swich", "swittch"],
    "synchronized": ["synchroized", "syncrhonized", "sychronized"],
    "this": ["tis", "thi", "thsi"],
    "throw": ["thow", "thrw", "trow"],
    "throws": ["trows", "thows", "thros"],
    "transient": ["transiet", "trasient", "trnsient"],
    "try": ["tyr", "tr", "trry"],
    "void": ["vod", "vid", "voiid"],
    "volatile": ["voltile", "volatie", "voaltile"],
    "while": ["whle", "wile", "wihle"],
    "{": [""],
    "}": [""],
    "(": [""],
    ")": [""],
    "[": [""],
    "]": [""],
    ";": [""],
    ",": [""],
    ".": [""],
    "+": [""],
    "-": [""],
    "*": [""],
    "/": [""],
    "&": [""],
    "|": [""],
    "^": [""],
    "%": [""],
    "!": [""],
    "~": [""],
    "?": [""],
    ":": [""],
    "<": [""],
    ">": [""],
    "=": [""],
    "'": [""],
    "\"": [""],
    "//": [""]
}

def corrupt_java_code(java_code):
    """
    This function randomly "corrupts" a line from the given Java code by replacing a keyword with a misspelled version.
    Or by removing a special symbol.

    Parameters:
    java_code (str): The input Java code as a string.

    Returns:
    str: The corrupted Java code as a string.
    """
    code_lines = java_code.split('\n')
    code_line_indices = list(range(len(code_lines)))
    should_corrupt = True

    possible_tokens = []
    pattern = r'\w+|\S'

    while should_corrupt:
        random_index = random.choice(code_line_indices)
        line = code_lines[random_index]

        if line.lstrip().startswith("//"):
            continue

        tokens = re.findall(pattern, line)

        for token in tokens:
            if token in java_keywords_misspellings:
                possible_tokens.append(token)

        if len(possible_tokens) == 0:
            continue

        token_to_replace = random.choice(possible_tokens)
        replace_with = random.choice(java_keywords_misspellings[token_to_replace])

        code_lines[random_index] = line.replace(token_to_replace, replace_with, 1)
        should_corrupt = False

    return '\n'.join(code_lines)



# Define misspellings for Python keywords and symbols
python_keywords_misspellings = {
    "and": ["nad", "adn", "andd"],
    "as": ["sa", "ass", "a"],
    "assert": ["assret", "asert", "assertt"],
    "break": ["brek", "brak", "breeak"],
    "class": ["clss", "cllas", "clsss"],
    "continue": ["conitnue", "cntinue", "contnue"],
    "def": ["dfe", "deef", "defe"],
    "del": ["dle", "del", "dell"],
    "elif": ["elif", "elif", "elef"],
    "else": ["els", "elese", "elsee"],
    "except": ["excep", "ecept", "excpt"],
    "finally": ["fianlly", "finaly", "finalyl"],
    "for": ["fro", "forr", "foor"],
    "from": ["fom", "frm", "fromm"],
    "global": ["gloabl", "globla", "gloal"],
    "if": ["f", "iff", "fi"],
    "import": ["improt", "impor", "importt"],
    "in": ["n", "in", "in "],
    "is": ["si", "is", "s"],
    "lambda": ["lambd", "lambaa", "lambd"],
    "nonlocal": ["nonlcao", "nonlocl", "nnonlocal"],
    "not": ["nt", "not", "nott"],
    "or": ["or", "o", "orr"],
    "pass": ["psas", "pasa", "pas"],
    "raise": ["raze", "raies", "raies"],
    "return": ["retun", "retur", "rturn"],
    "try": ["tyr", "tr", "trry"],
    "while": ["whle", "wile", "wihle"],
    "with": ["wiht", "wih", "withh"],
    "yield": ["yeld", "yeild", "yld"],
    "#": [""],
    "(": [""],
    ")": [""],
    "[": [""],
    "]": [""],
    "{": [""],
    "}": [""],
    ";": [""],
    ",": [""],
    ".": [""],
    "+": [""],
    "-": [""],
    "*": [""],
    "/": [""],
    "&": [""],
    "|": [""],
    "^": [""],
    "%": [""],
    "!": [""],
    "~": [""],
    "?": [""],
    ":": [""],
    "<": [""],
    ">": [""],
    "=": [""],
    "'": [""],
    "\"": [""]
}


def corrupt_python_code(python_code):
    """
    This function randomly "corrupts" a line from the given Python code by replacing a keyword with a misspelled version.
    Or by removing a special symbol.

    Parameters:
    java_code (str): The input Python code as a string.

    Returns:
    str: The corrupted Python code as a string.
    """
    code_lines = python_code.split('\n')
    code_line_indices = list(range(len(code_lines)))
    should_corrupt = True

    possible_tokens = []
    pattern = r'\w+|\S'

    while should_corrupt:
        random_index = random.choice(code_line_indices)
        line = code_lines[random_index]

        if line.lstrip().startswith("#"):
            continue

        tokens = re.findall(pattern, line)

        for token in tokens:
            if token in python_keywords_misspellings:
                possible_tokens.append(token)

        if len(possible_tokens) == 0:
            continue

        token_to_replace = random.choice(possible_tokens)
        replace_with = random.choice(python_keywords_misspellings[token_to_replace])

        code_lines[random_index] = line.replace(token_to_replace, replace_with, 1)
        should_corrupt = False

    return '\n'.join(code_lines)
