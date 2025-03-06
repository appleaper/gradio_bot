

import logging
import copy
import datrie
import math
import os
import re
import string
import sys
from hanziconv import HanziConv
import jieba
from config import conf_yaml

trie_dir_path  = conf_yaml['rag']['trie_dir_path']
class RagTokenizer:
    def key_(self, line):
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/key_')
        '''将输入的字符串转换为小写，接着进行 UTF - 8 编码，最后去掉编码结果首尾的部分字符。'''
        return str(line.lower().encode("utf-8"))[2:-1]

    def rkey_(self, line):
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/rkey_')
        '''先将输入字符串反转并转换为小写，然后在前面加上 "DD"，再进行 UTF - 8 编码，最后去掉编码结果首尾的部分字符。'''
        return str(("DD" + (line[::-1].lower())).encode("utf-8"))[2:-1]

    def loadDict_(self, fnm):
        '''
        loadDict_ 方法的核心目的是从文件中读取数据，构建并更新字典树，最后将字典树保存到缓存文件中，同时在处理过程中记录关键信息和异常情况。
        更新过程中会计算一个分值，分数高才能更新到字典树中
        '''
        print('rag/nlp/rag_tokenizer.py/RagTokenizer/loadDict_')
        logging.info(f"[HUQIE]:Build trie from {fnm}")
        try:
            of = open(fnm, "r", encoding='utf-8')
            while True:
                line = of.readline()
                if not line:
                    break
                line = re.sub(r"[\r\n]+", "", line)
                line = re.split(r"[ \t]", line)
                k = self.key_(line[0])
                F = int(math.log(float(line[1]) / self.DENOMINATOR) + .5)
                if k not in self.trie_ or self.trie_[k][0] < F:
                    self.trie_[self.key_(line[0])] = (F, line[2])
                self.trie_[self.rkey_(line[0])] = 1

            dict_file_cache = fnm + ".trie"
            logging.info(f"[HUQIE]:Build trie cache to {dict_file_cache}")
            self.trie_.save(dict_file_cache)
            of.close()
        except Exception:
            logging.exception(f"[HUQIE]:Build trie {fnm} failed")

    def __init__(self, debug=False):
        self.DEBUG = debug
        self.DENOMINATOR = 1000000
        self.DIR_ = os.path.join(trie_dir_path, "huqie")
        self.SPLIT_CHAR = r"([ ,\.<>/?;:'\[\]\\`!@#$%^&*\(\)\{\}\|_+=《》，。？、；‘’：“”【】~！￥%……（）——-]+|[a-z\.-]+|[0-9,\.-]+)"

        trie_file_name = self.DIR_ + ".txt.trie"
        # check if trie file existence
        if os.path.exists(trie_file_name):
            try:
                # load trie from file
                self.trie_ = datrie.Trie.load(trie_file_name)
                return
            except Exception:
                # fail to load trie from file, build default trie
                logging.exception(f"[HUQIE]:Fail to load trie file {trie_file_name}, build the default trie file")
                self.trie_ = datrie.Trie(string.printable)
        else:
            # file not exist, build default trie
            logging.info(f"[HUQIE]:Trie file {trie_file_name} not found, build the default trie file")
            self.trie_ = datrie.Trie(string.printable)

        # load data from dict file and save to trie file
        self.loadDict_(self.DIR_ + ".txt")

    def loadUserDict(self, fnm):
        print('rag/nlp/rag_tokenizer.py/RagTokenizer/loadUserDict')
        '''
        loadUserDict 方法的主要功能是尝试从指定文件对应的 .trie 缓存文件中加载字典树（Trie）。如果加载成功，则直接使用加载的字典树；
        如果加载失败，会创建一个默认的字典树，然后从指定文件中加载数据并更新字典树。
        '''
        try:
            self.trie_ = datrie.Trie.load(fnm + ".trie")
            return
        except Exception:
            self.trie_ = datrie.Trie(string.printable)
        self.loadDict_(fnm)

    def addUserDict(self, fnm):
        self.loadDict_(fnm)

    def _strQ2B(self, ustring):
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/_strQ2B')
        """
        Convert full-width characters to half-width characters
        该函数 _strQ2B 将全角字符转换为半角字符。具体步骤如下：
        初始化空字符串 rstring。
        遍历输入字符串 ustring 的每个字符 uchar。
        获取字符的 Unicode 编码 inside_code。
        如果字符是全角空格（Unicode 0x3000），转换为半角空格（Unicode 0x0020）；否则，将编码减去 0xfee0。
        检查转换后的编码是否在半角字符范围内（0x0020 到 0x7e），如果是则添加到结果字符串，否则保留原字符。
        返回最终的结果字符串。
        """
        """Convert full-width characters to half-width characters"""
        rstring = ""
        for uchar in ustring:
            inside_code = ord(uchar)
            if inside_code == 0x3000:
                inside_code = 0x0020
            else:
                inside_code -= 0xfee0
            if inside_code < 0x0020 or inside_code > 0x7e:  # After the conversion, if it's not a half-width character, return the original character.
                rstring += uchar
            else:
                rstring += chr(inside_code)
        return rstring

    def _tradi2simp(self, line):
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/_tradi2simp')
        '''
        _tradi2simp 方法是一个实例方法，其主要功能是将输入的包含繁体中文的字符串转换为简体中文。它借助 HanziConv 库中的 toSimplified 函数来实现这一转换。
        '''
        return HanziConv.toSimplified(line)

    def dfs_(self, chars, s, preTks, tkslist):
        '''
        dfs_ 方法实现了一个深度优先搜索（DFS）算法，主要用于对输入的字符序列 chars 进行分词处理，找出所有可能的分词组合，并将这些组合存储在 tkslist 列表中。
        在搜索过程中，会利用字典树 self.trie_ 来判断可能的分词是否存在于字典中，同时采用了一些剪枝策略来减少不必要的搜索。

        示例说明
        假设输入的字符序列 chars = ['你', '好', '世', '界']，字典树 self.trie_ 中包含键 '你好' 和 '世界'。
        初始调用 dfs_(chars, 0, [], tkslist)，s = 0，res = 0。
        第一个剪枝条件不满足，S = 1。
        进入循环：
        当 e = 1 时，t = '你'，k = self.key_('你')，若 k 不在字典树中，继续。
        当 e = 2 时，t = '你好'，k = self.key_('你好')，k 在字典树中，复制 preTks 到 pretks，添加 ('你好', self.trie_['你好']) 到 pretks，
        递归调用 dfs_(chars, 2, pretks, tkslist)。
        在新的递归调用中，重复上述步骤，处理 '世界' 分词。
        最终，tkslist 中会存储所有可能的分词组合，如 [('你好', ...), ('世界', ...)]。
        '''
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/dfs')
        res = s
        # if s > MAX_L or s>= len(chars):
        if s >= len(chars):
            tkslist.append(preTks)
            return res

        # pruning
        S = s + 1
        if s + 2 <= len(chars):
            t1, t2 = "".join(chars[s:s + 1]), "".join(chars[s:s + 2])
            if self.trie_.has_keys_with_prefix(self.key_(t1)) and not self.trie_.has_keys_with_prefix(
                    self.key_(t2)):
                S = s + 2
        if len(preTks) > 2 and len(
                preTks[-1][0]) == 1 and len(preTks[-2][0]) == 1 and len(preTks[-3][0]) == 1:
            t1 = preTks[-1][0] + "".join(chars[s:s + 1])
            if self.trie_.has_keys_with_prefix(self.key_(t1)):
                S = s + 2

        ################
        for e in range(S, len(chars) + 1):
            t = "".join(chars[s:e])
            k = self.key_(t)

            if e > s + 1 and not self.trie_.has_keys_with_prefix(k):
                break

            if k in self.trie_:
                pretks = copy.deepcopy(preTks)
                if k in self.trie_:
                    pretks.append((t, self.trie_[k]))
                else:
                    pretks.append((t, (-12, '')))
                res = max(res, self.dfs_(chars, e, pretks, tkslist))

        if res > s:
            return res

        t = "".join(chars[s:s + 1])
        k = self.key_(t)
        if k in self.trie_:
            preTks.append((t, self.trie_[k]))
        else:
            preTks.append((t, (-12, '')))

        return self.dfs_(chars, s + 1, preTks, tkslist)

    def freq(self, tk):
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/freq')
        '''
        这段代码定义了一个名为 freq 的实例方法，其主要功能是根据输入的分词 tk，从字典树 self.trie_ 中查找该分词对应的频率值。
        若该分词存在于字典树中，会根据存储的相关信息计算并返回其频率；若不存在，则返回频率为 0。
        '''
        k = self.key_(tk)
        if k not in self.trie_:
            return 0
        return int(math.exp(self.trie_[k][0]) * self.DENOMINATOR + 0.5)

    def tag(self, tk):
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/tag')
        '''
        tag 方法是一个实例方法，其主要作用是根据输入的分词 tk，从字典树 self.trie_ 中查找该分词对应的标签信息。
        如果该分词存在于字典树中，则返回对应的标签；若不存在，则返回空字符串。
        '''
        k = self.key_(tk)
        if k not in self.trie_:
            return ""
        return self.trie_[k][1]

    def score_(self, tfts):
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/score_')
        '''
        score_ 方法的主要功能是根据输入的分词及其频率和标签信息，计算一个得分值，并返回分词列表和该得分值。该得分综合考虑了分词的总频率、长词比例以及分词数量等因素。
        '''
        B = 30
        F, L, tks = 0, 0, []
        for tk, (freq, tag) in tfts:
            F += freq
            L += 0 if len(tk) < 2 else 1
            tks.append(tk)
        #F /= len(tks)
        L /= len(tks)
        logging.debug("[SC] {} {} {} {} {}".format(tks, len(tks), L, F, B / len(tks) + L + F))
        return tks, B / len(tks) + L + F

    def sortTks_(self, tkslist):
        '''
        sortTks_ 方法的主要功能是对输入的分词组合列表 tkslist 进行排序。它会遍历列表中的每个分词组合，为每个组合计算一个得分，
        然后根据这些得分对所有组合进行降序排序，最终返回排序后的结果。
        '''
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/sortTks_')
        res = []
        for tfts in tkslist:
            tks, s = self.score_(tfts)
            res.append((tks, s))
        return sorted(res, key=lambda x: x[1], reverse=True)

    def merge_(self, tks):
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/merge_')
        '''
        通过遍历分词列表，尝试将相邻的分词合并成更长的词，同时考虑合并后的词是否包含分隔字符以及其出现频率，最终返回合并后的分词字符串。
        这种合并操作可以在一定程度上提高分词结果的合理性和语义连贯性。
        '''
        # if split chars is part of token
        res = []
        tks = re.sub(r"[ ]+", " ", tks).split()
        s = 0
        while True:
            if s >= len(tks):
                break
            E = s + 1
            for e in range(s + 2, min(len(tks) + 2, s + 6)):
                tk = "".join(tks[s:e])
                if re.search(self.SPLIT_CHAR, tk) and self.freq(tk):
                    E = e
            res.append("".join(tks[s:E]))
            s = E

        return " ".join(res)

    def maxForward_(self, line):
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/maxForward_')
        '''
        maxForward_ 方法通过正向最大匹配算法对输入的字符串进行分词，尽可能匹配词典中最长的词。它使用 self.trie_ 作为词典，
        根据子串是否在词典中来确定分词结果，并最终调用 self.score_ 方法对分词结果进行打分。
        '''
        res = []
        s = 0
        while s < len(line):
            e = s + 1
            t = line[s:e]
            while e < len(line) and self.trie_.has_keys_with_prefix(
                    self.key_(t)):
                e += 1
                t = line[s:e]

            while e - 1 > s and self.key_(t) not in self.trie_:
                e -= 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s = e

        return self.score_(res)

    def maxBackward_(self, line):
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/maxBackward_')
        '''
        maxBackward_ 方法实现了一种最大逆向匹配的分词算法，用于对输入的字符串 line 进行分词处理。该算法从字符串的末尾开始，
        逆向尝试找出尽可能长的存在于字典树 self.trie_ 中的分词，将分词结果存储在列表中，最后调用 self.score_ 方法为分词结果计算得分并返回。
        '''
        res = []
        s = len(line) - 1
        while s >= 0:
            e = s + 1
            t = line[s:e]
            while s > 0 and self.trie_.has_keys_with_prefix(self.rkey_(t)):
                s -= 1
                t = line[s:e]

            while s + 1 < e and self.key_(t) not in self.trie_:
                s += 1
                t = line[s:e]

            if self.key_(t) in self.trie_:
                res.append((t, self.trie_[self.key_(t)]))
            else:
                res.append((t, (0, '')))

            s -= 1

        return self.score_(res[::-1])

    def english_normalize_(self, tks):
        return tks

    def tokenize(self, line):
        line = re.sub(r"\W+", " ", line)
        line = self._strQ2B(line).lower()
        line = self._tradi2simp(line)
        zh_num = len([1 for c in line if is_chinese(c)])
        if zh_num == 0:
            return " ".join(jieba.lcut(line))

        arr = re.split(self.SPLIT_CHAR, line)
        res = []
        for L in arr:
            if len(L) < 2 or re.match(
                    r"[a-z\.-]+$", L) or re.match(r"[0-9\.-]+$", L):
                res.append(L)
                continue

            # use maxforward for the first time
            tks, s = self.maxForward_(L)
            tks1, s1 = self.maxBackward_(L)
            if self.DEBUG:
                logging.debug("[FW] {} {}".format(tks, s))
                logging.debug("[BW] {} {}".format(tks1, s1))

            i, j, _i, _j = 0, 0, 0, 0
            same = 0

            while i + same < len(tks1) and j + same < len(tks) and tks1[i + same] == tks[j + same]:
                same += 1
            if same > 0:
                res.append(" ".join(tks[j: j + same]))
            _i = i + same
            _j = j + same
            j = _j + 1
            i = _i + 1

            while i < len(tks1) and j < len(tks):
                tk1, tk = "".join(tks1[_i:i]), "".join(tks[_j:j])
                if tk1 != tk:
                    if len(tk1) > len(tk):
                        j += 1
                    else:
                        i += 1
                    continue

                if tks1[i] != tks[j]:
                    i += 1
                    j += 1
                    continue
                # backward tokens from_i to i are different from forward tokens from _j to j.
                tkslist = []
                self.dfs_("".join(tks[_j:j]), 0, [], tkslist)
                res.append(" ".join(self.sortTks_(tkslist)[0][0]))

                same = 1
                while i + same < len(tks1) and j + same < len(tks) and tks1[i + same] == tks[j + same]:
                    same += 1
                res.append(" ".join(tks[j: j + same]))
                _i = i + same
                _j = j + same
                j = _j + 1
                i = _i + 1

            if _i < len(tks1):
                assert _j < len(tks)
                assert "".join(tks1[_i:]) == "".join(tks[_j:])
                tkslist = []
                self.dfs_("".join(tks[_j:]), 0, [], tkslist)
                res.append(" ".join(self.sortTks_(tkslist)[0][0]))

        res = " ".join(self.english_normalize_(res))
        logging.debug("[TKS] {}".format(self.merge_(res)))
        return self.merge_(res)

    def fine_grained_tokenize(self, tks):
        # print('rag/nlp/rag_tokenizer.py/RagTokenizer/fine_grained_tokenize')
        '''
        主要功能是对已经经过初步分词处理的结果 tks 进行更细粒度的分词操作。该方法会根据输入中中文字符的占比情况采用不同的处理策略，
        对于英文和数字等也有相应的处理逻辑，最后对处理结果中的英文单词进行归一化处理。
        '''
        tks = tks.split()
        zh_num = len([1 for c in tks if c and is_chinese(c[0])])
        if zh_num < len(tks) * 0.2:
            res = []
            for tk in tks:
                res.extend(tk.split("/"))
            return " ".join(res)

        res = []
        for tk in tks:
            if len(tk) < 3 or re.match(r"[0-9,\.-]+$", tk):
                res.append(tk)
                continue
            tkslist = []
            if len(tk) > 10:
                tkslist.append(tk)
            else:
                self.dfs_(tk, 0, [], tkslist)
            if len(tkslist) < 2:
                res.append(tk)
                continue
            stk = self.sortTks_(tkslist)[1][0]
            if len(stk) == len(tk):
                stk = tk
            else:
                if re.match(r"[a-z\.-]+$", tk):
                    for t in stk:
                        if len(t) < 3:
                            stk = tk
                            break
                    else:
                        stk = " ".join(stk)
                else:
                    stk = " ".join(stk)

            res.append(stk)

        return " ".join(self.english_normalize_(res))


def is_chinese(s):
    # print('rag/nlp/rag_tokenizer.py/is_chinese')
    # 判断是否是汉字
    if s >= u'\u4e00' and s <= u'\u9fa5':
        return True
    else:
        return False


def is_number(s):
    print('rag/nlp/rag_tokenizer.py/is_number')
    if s >= u'\u0030' and s <= u'\u0039':
        return True
    else:
        return False


def is_alphabet(s):
    print('rag/nlp/rag_tokenizer.py/is_alphabet')
    if (s >= u'\u0041' and s <= u'\u005a') or (
            s >= u'\u0061' and s <= u'\u007a'):
        return True
    else:
        return False


def naiveQie(txt):
    print('rag/nlp/rag_tokenizer.py/naiveQie')
    tks = []
    for t in txt.split():
        if tks and re.match(r".*[a-zA-Z]$", tks[-1]
                            ) and re.match(r".*[a-zA-Z]$", t):
            tks.append(" ")
        tks.append(t)
    return tks


tokenizer = RagTokenizer()
tokenize = tokenizer.tokenize
fine_grained_tokenize = tokenizer.fine_grained_tokenize
tag = tokenizer.tag
freq = tokenizer.freq
loadUserDict = tokenizer.loadUserDict
addUserDict = tokenizer.addUserDict
tradi2simp = tokenizer._tradi2simp
strQ2B = tokenizer._strQ2B

if __name__ == '__main__':
    tknzr = RagTokenizer(debug=True)
    # huqie.addUserDict("/tmp/tmp.new.tks.dict")
    tks = tknzr.tokenize(
        "哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈哈")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "公开征求意见稿提出，境外投资者可使用自有人民币或外汇投资。使用外汇投资的，可通过债券持有人在香港人民币业务清算行及香港地区经批准可进入境内银行间外汇市场进行交易的境外人民币业务参加行（以下统称香港结算行）办理外汇资金兑换。香港结算行由此所产生的头寸可到境内银行间外汇市场平盘。使用外汇投资的，在其投资的债券到期或卖出后，原则上应兑换回外汇。")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "多校划片就是一个小区对应多个小学初中，让买了学区房的家庭也不确定到底能上哪个学校。目的是通过这种方式为学区房降温，把就近入学落到实处。南京市长江大桥")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "实际上当时他们已经将业务中心偏移到安全部门和针对政府企业的部门 Scripts are compiled and cached aaaaaaaaa")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("虽然我不怎么玩")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("蓝月亮如何在外资夹击中生存,那是全宇宙最有意思的")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "涡轮增压发动机num最大功率,不像别的共享买车锁电子化的手段,我们接过来是否有意义,黄黄爱美食,不过，今天阿奇要讲到的这家农贸市场，说实话，还真蛮有特色的！不仅环境好，还打出了")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("这周日你去吗？这周日你有空吗？")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize("Unity3D开发经验 测试开发工程师 c++双11双11 985 211 ")
    logging.info(tknzr.fine_grained_tokenize(tks))
    tks = tknzr.tokenize(
        "数据分析项目经理|数据分析挖掘|数据分析方向|商品数据分析|搜索数据分析 sql python hive tableau Cocos2d-")
    logging.info(tknzr.fine_grained_tokenize(tks))
    if len(sys.argv) < 2:
        sys.exit()
    tknzr.DEBUG = False
    tknzr.loadUserDict(sys.argv[1])
    of = open(sys.argv[2], "r")
    while True:
        line = of.readline()
        if not line:
            break
        logging.info(tknzr.tokenize(line))
    of.close()
