import re
from cati.utils.cati_config import *


class Converter:

    exDict = {}

    def __init__(self, raw_bytes=False):
        """Takes in input the text file in which is saved the dictionary to translate in OPCode"""
        if not raw_bytes:
            with open(DICTIONARY, encoding='utf-8') as f:
                for line in f:
                    (key, val) = line.split(" -> ")
                    val = val.strip("\n")
                    self.exDict[key] = val
        else:
            with open(DICTIONARYrawbytes, encoding='utf-8') as f:
                for line in f:
                    (key, val) = line.split(" -> ")
                    val = val.strip("\n")
                    self.exDict[key] = val

    def __str__(self):
        return f"These {self.exDict} are the words that will be converted"

    def encoder(self, content):
        """For every word in the converting dictionary check if in the smali text
        there is a correspondence, if true tranlate it in the opportune val of the key"""
        for word, opcode in self.exDict.items():
            content = re.sub(word, opcode, content)
        return content

    def methods_indexes(self, content):
        start_ch = self.exDict.get('.method')
        end_ch = self.exDict.get('.end method')
        methods_index = []
        st = None
        for i, ch in enumerate(content):
            if ch == start_ch:
                st = i
            elif ch == end_ch:
                if st is not None:
                    methods_index.append((st, i))
                    st = None
        return methods_index
