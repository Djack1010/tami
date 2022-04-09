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

    def analyse_content(self, smali_paths, family, byte_format=False):
        general_content = b'' if byte_format else ""
        smali_k = {}
        for smali in smali_paths:
            class_name = smali.replace(f'{DECOMPILED}/{family}', "")
            fr = open(smali, "r")
            content = self.encoder(fr.read())
            fr.close()

            if byte_format:
                content = content.encode()

            # saving number of character and content
            num_character = len(content)
            if byte_format:
                meth_indexes = self.methods_indexes(content, byte_format)
                smali_k[class_name] = {'len': num_character, 'meth': meth_indexes}
                general_content = b''.join([general_content, content])
            else:
                smali_k[class_name] = num_character
                general_content += content
        return smali_k, general_content

    def methods_indexes(self, content, byte_format=False):
        start_ch = self.exDict.get('.method').encode() if byte_format else self.exDict.get('.method')
        end_ch = self.exDict.get('.end method').encode() if byte_format else self.exDict.get('.end method')
        methods_index = []
        st = None
        if byte_format:
            iterable_bytes = [content[i:i+1] for i in range(len(content))]
            for i in range(len(iterable_bytes)-1):
                ch = b''.join([iterable_bytes[i], iterable_bytes[i+1]])
                if ch == start_ch:
                    st = i
                elif ch == end_ch:
                    if st is not None:
                        methods_index.append((st, i))
                        st = None
        else:
            for i, ch in enumerate(content):
                if ch == start_ch:
                    st = i
                elif ch == end_ch:
                    if st is not None:
                        methods_index.append((st, i))
                        st = None
        return methods_index
