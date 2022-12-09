import re

from basic.models import Document


def increase_char(char):
    if char == "Z":
        return "A",True
    else:
        return chr(ord(char) + 1),False

def increase_string(string):
    result_reversed = ""
    increase_next = True
    for char in reversed(string):
        if increase_next:
            new_char, increase_next = increase_char(char)
            result_reversed+=new_char
        else:
            result_reversed+=char
    result = "".join(reversed(result_reversed))
    return result

def increase_id(id):
    hs = int(len(id)/2)
    char_part,number_part = id[:hs],id[hs:]
    if int(number_part) == 999:
        new_number_part = str(0).zfill(3)
        new_char_part = increase_string(char_part)
    else:
        new_number_part = str(int(number_part)+1).zfill(3)
        new_char_part = char_part
    return new_char_part+new_number_part

def get_human_readable_id():
    if Document.objects.filter(human_readable_id="").count() == 0:
        human_readable_id = "AAA000"
    else:
        last_entry = Document.objects.latest("human_readable_id")
        human_readable_id = last_entry.human_readable_id
    next_human_readable_id = increase_id(human_readable_id)
    return next_human_readable_id

def is_human_readable_id(id):
    return bool(re.match(r"[A-Z]{3}[0-9]{3}",id))