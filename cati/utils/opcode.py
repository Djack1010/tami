import re
from cati.utils.config import *


class Converter:

    exDict = {}

    def __init__(self):
        """Takes in input the text file in which is saved the dictionary to translate in OPCode"""
        with open(DICTIONARY, encoding='utf-8') as f:
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
