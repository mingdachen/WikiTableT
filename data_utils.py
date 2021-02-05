import os
import pickle
import json
import statistics

import numpy as np

from glob import glob
from subword_nmt.apply_bpe import BPE, read_vocabulary
from decorators import auto_init_args
from config import UNK_IDX, UNK_WORD, BOS_IDX, EOS_IDX, \
    BOC_IDX, BOV_IDX, BOQK_IDX, BOQV_IDX, SUBSEC_IDX, \
    EOC_IDX, MAX_VALUE_LEN, MASK_IDX


class DataHolder:
    @auto_init_args
    def __init__(self, train_data, dev_data, test_data, vocab, bpe):
        self.inv_vocab = {i: w for w, i in vocab.items()}


class DataProcessor:
    @auto_init_args
    def __init__(self, train_path, dev_path, test_path,
                 wikidata_path, infobox_path,
                 bpe_vocab, bpe_codes, experiment):
        self.expe = experiment
        bpe_vocab_dict = read_vocabulary(open(bpe_vocab), 50)
        self.bpe = BPE(open(bpe_codes), -1, "@@", bpe_vocab_dict, None)

    def process(self):

        vocab = {UNK_WORD: UNK_IDX, "<bos>": BOS_IDX, "<eos>": EOS_IDX,
                 "<boc>": BOC_IDX, "<bov>": BOV_IDX, "<boqk>": BOQK_IDX,
                 "<boqv>": BOQV_IDX, "<subsec>": SUBSEC_IDX, "<eoc>": EOC_IDX,
                 "<mask>": MASK_IDX}

        with open(self.bpe_vocab) as fp:
            for line in fp:
                w, f = line.strip().split()
                if int(f) > 50:
                    vocab[w] = len(vocab)

        self.expe.log.info("vocab size: {}".format(len(vocab)))

        train_data, dev_data, test_data = self._load_sent_and_data(
            self.train_path, self.dev_path, self.test_path,
            self.wikidata_path, self.infobox_path, vocab)

        def cal_stats(data):
            data_unk_count = 0
            data_total_count = 0
            text_unk_count = 0
            text_total_count = 0
            leng_text = []
            leng_hyp_data = []
            leng_wiki_data = []
            leng_sec_data = []
            leng_doc_data = []
            for d in data:
                assert len(d["hyperlink_data"]) == 5, \
                    "d['hyperlink_data'] = {} != 5"\
                    .format(len(d["hyperlink_data"]))
                leng_text.append(len(d["text"]))
                leng_hyp_data.append(len(d["hyperlink_data"][0]))
                leng_wiki_data.append(len(d["wikidata"][0]))
                leng_sec_data.append(len(d["sec_title"][0]))
                leng_doc_data.append(len(d["doc_title"][0]))
                for word in d["hyperlink_data"][0]:
                    if word == UNK_IDX:
                        data_unk_count += 1
                    data_total_count += 1
                for word in d["wikidata"][0]:
                    if word == UNK_IDX:
                        data_unk_count += 1
                    data_total_count += 1
                for word in d["text"]:
                    if word == UNK_IDX:
                        text_unk_count += 1
                    text_total_count += 1
            return (data_unk_count, data_total_count,
                    data_unk_count / data_total_count * 100
                    if data_total_count else 0), \
                   (text_unk_count, text_total_count,
                    text_unk_count / text_total_count * 100
                    if text_total_count else 0), \
                   (len(leng_hyp_data), max(leng_hyp_data),
                    min(leng_hyp_data), sum(leng_hyp_data) / len(leng_hyp_data)
                    if len(leng_hyp_data) else 0,
                    statistics.median(leng_hyp_data)
                    ), \
                   (len(leng_wiki_data), max(leng_wiki_data),
                    min(leng_wiki_data),
                    sum(leng_wiki_data) / len(leng_wiki_data)
                    if len(leng_wiki_data) else 0,
                    statistics.median(leng_wiki_data)), \
                   (len(leng_sec_data),
                    max(leng_sec_data), min(leng_sec_data),
                    sum(leng_sec_data) / len(leng_sec_data)
                    if len(leng_sec_data) else 0,
                    statistics.median(leng_sec_data)), \
                   (len(leng_doc_data), max(leng_doc_data),
                    min(leng_doc_data), sum(leng_doc_data) / len(leng_doc_data)
                    if len(leng_doc_data) else 0,
                    statistics.median(leng_doc_data)), \
                   (len(leng_text), max(leng_text),
                    min(leng_text), sum(leng_text) / len(leng_text)
                    if len(leng_text) else 0,
                    statistics.median(leng_text)
                    )

        data_unk_stats, text_unk_stats, \
            data_hyp_len_stats, text_wiki_len_stats, \
            text_sec_len_stats, leng_doc_data, \
            text_len_stats = cal_stats(train_data)
        self.expe.log.info("#train hyp data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}, median len: {:.2f}"
                           .format(*data_hyp_len_stats))

        self.expe.log.info("#train wiki data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}, median len: {:.2f}"
                           .format(*text_wiki_len_stats))

        self.expe.log.info("#train sec data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}, median len: {:.2f}"
                           .format(*text_sec_len_stats))

        self.expe.log.info("#train doc data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}, median len: {:.2f}"
                           .format(*leng_doc_data))

        self.expe.log.info("#train text: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}, median len: {:.2f}"
                           .format(*text_len_stats))

        self.expe.log.info(
            "#train data unk: {}, {}, {:.4f}%".format(*data_unk_stats))
        self.expe.log.info(
            "#train text unk: {}, {}, {:.4f}%".format(*text_unk_stats))

        self.expe.log.info("*" * 50)

        data_unk_stats, text_unk_stats, \
            data_hyp_len_stats, text_wiki_len_stats, \
            text_sec_len_stats, leng_doc_data, \
            text_len_stats = cal_stats(dev_data)

        self.expe.log.info("#dev hyp data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}, median len: {:.2f}"
                           .format(*data_hyp_len_stats))

        self.expe.log.info("#dev wiki data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}, median len: {:.2f}"
                           .format(*text_wiki_len_stats))

        self.expe.log.info("#dev sec data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}, median len: {:.2f}"
                           .format(*text_sec_len_stats))

        self.expe.log.info("#dev doc data: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}, median len: {:.2f}"
                           .format(*leng_doc_data))

        self.expe.log.info("#dev text: {}, max len: {}, "
                           "min len: {}, avg len: {:.2f}, median len: {:.2f}"
                           .format(*text_len_stats))

        self.expe.log.info(
            "#dev data unk: {}, {}, {:.4f}%".format(*data_unk_stats))
        self.expe.log.info(
            "#dev text unk: {}, {}, {:.4f}%".format(*text_unk_stats))

        self.expe.log.info("*" * 50)
        data = DataHolder(
            train_data=np.array(train_data),
            dev_data=np.array(dev_data),
            test_data=None,
            vocab=vocab,
            bpe=self.bpe)

        return data

    def _load_sent_and_data(self, train_path, dev_path,
                            test_path, wikidata_path, infobox_path, vocab):
        infobox = {}
        trunc_max_n_data = 0
        max_n_keys = 0
        max_n_data = 0
        max_datatype_count = 0
        n_skip = 0

        if infobox_path is not None:
            with open(infobox_path) as fp:
                for nline, dataline in enumerate(fp):
                    if nline and nline % 50000 == 0:
                        self.expe.log.info(
                            "loading infobox #line: {}".format(nline))
                    if dataline.strip():
                        datajson = json.loads(dataline.strip())

                        datalist = []
                        datapos = []
                        datatype = []
                        datamask = []
                        datasrc = []
                        datatype_count = 0

                        max_n_keys = max(max_n_keys, len(datajson["infobox"]))
                        for key in datajson["infobox"]:
                            all_data_for_key = datajson["infobox"][key]
                            max_n_data = max(max_n_data, len(all_data_for_key))
                            trunc_max_n_data = max(trunc_max_n_data,
                                                   len(all_data_for_key))

                            prop_value = [vocab["<boc>"]]
                            prop_mask = [vocab["<boc>"]]
                            prop_src = ["<boc>"]

                            temp_value = [vocab.get(w, 0) for w in
                                          key.split()[:MAX_VALUE_LEN]]
                            prop_value += temp_value
                            prop_mask += [vocab["<mask>"]] * len(temp_value)
                            prop_src += key.split()[:MAX_VALUE_LEN]

                            prop_value += [vocab["<bov>"]]
                            prop_mask += [vocab["<bov>"]]
                            prop_src += ["<bov>"]

                            temp_value = [vocab.get(w, 0) for w in
                                          all_data_for_key.split()
                                          [:MAX_VALUE_LEN]]
                            prop_value += temp_value
                            prop_mask += [vocab["<mask>"]] * len(temp_value)
                            prop_src += \
                                all_data_for_key.split()[:MAX_VALUE_LEN]

                            if len(datalist) + len(prop_value) > 350 or \
                                    datatype_count > 300:
                                continue
                            datamask += prop_mask
                            datalist += prop_value
                            datapos += list(range(len(prop_value)))
                            datatype += [datatype_count] * len(prop_value)
                            datasrc += prop_src
                            datatype_count += 1
                        max_datatype_count = \
                            max(max_datatype_count, datatype_count)
                        assert len(datalist) == len(datamask), \
                            "{} != {}".format(len(datalist), len(datamask))
                        assert len(datalist) == len(datasrc), \
                            "{} != {}".format(len(datalist), len(datasrc))
                        if datalist:
                            infobox[datajson["title"]] = \
                                (datalist, datapos, datatype,
                                 datamask, datasrc)
                        else:
                            n_skip += 1
        self.expe.log.info("loaded #infobox entries: {}, "
                           "truncated max #data: {}, "
                           "non-truncated max #data: {}, "
                           "max #keys: {}, "
                           "max datatype count: {}, "
                           "#skip: {}"
                           .format(len(infobox), trunc_max_n_data,
                                   max_n_data, max_n_keys,
                                   max_datatype_count - 1,
                                   n_skip))

        dup = 0
        trunc_max_n_data = 0
        max_n_keys = 0
        max_n_data = 0
        max_n_qual = 0
        max_datatype_count = 0
        wikidata = {}
        n_skip = 0
        if wikidata_path is not None:
            with open(wikidata_path) as fp:
                for nline, dataline in enumerate(fp):
                    if nline and nline % 100000 == 0:
                        self.expe.log.info(
                            "loading wikidata #line: {}".format(nline))
                    if dataline.strip():
                        datajson = json.loads(dataline.strip())
                        datalist = []
                        datapos = []
                        datatype = []
                        datamask = []
                        datasrc = []
                        datatype_count = 0

                        max_n_keys = max(max_n_keys,
                                         len(datajson["wikidata_details"]))
                        for key in datajson["wikidata_details"]:
                            all_data_for_key = \
                                datajson["wikidata_details"][key]
                            max_n_data = max(max_n_data, len(all_data_for_key))
                            if self.expe.config.max_num_value is not None:
                                all_data_for_key = \
                                    all_data_for_key[:self.expe.config.max_num_value]
                            trunc_max_n_data = max(trunc_max_n_data,
                                                   len(all_data_for_key))
                            for l in all_data_for_key:
                                prop_value = [vocab["<boc>"]]
                                prop_mask = [vocab["<boc>"]]
                                prop_src = ["<boc>"]

                                temp_value = [vocab.get(w, 0) for w in
                                              key.split()[:MAX_VALUE_LEN]]
                                prop_value += temp_value
                                prop_mask += [vocab["<mask>"]] * \
                                    len(temp_value)
                                prop_src += key.split()[:MAX_VALUE_LEN]

                                prop_value += [vocab["<bov>"]]
                                prop_mask += [vocab["<bov>"]]
                                prop_src += ["<bov>"]

                                temp_value = \
                                    [vocab.get(w, 0) for w in
                                     l["data"].split()[:MAX_VALUE_LEN]]
                                prop_value += temp_value
                                prop_mask += [vocab["<mask>"]] * \
                                    len(temp_value)
                                prop_src += l["data"].split()[:MAX_VALUE_LEN]

                                if "qualifiers" in l:
                                    max_n_qual = max(max_n_qual,
                                                     len(l["qualifiers"]))
                                    for qual_key in l["qualifiers"]:
                                        prop_value += [vocab["<boqk>"]]
                                        prop_mask += [vocab["<boqk>"]]
                                        prop_src += ["<boqk>"]

                                        temp_value = \
                                            [vocab.get(w, 0) for w in
                                             qual_key.split()[:MAX_VALUE_LEN]]
                                        prop_value += temp_value
                                        prop_mask += [vocab["<mask>"]] * \
                                            len(temp_value)
                                        prop_src += \
                                            qual_key.split()[:MAX_VALUE_LEN]

                                        prop_value += [vocab["<boqv>"]]
                                        prop_mask += [vocab["<boqv>"]]
                                        prop_src += [vocab["<boqv>"]]

                                        temp_value = \
                                            [vocab.get(w, 0) for w in
                                             l["qualifiers"][qual_key]
                                             .split()[:MAX_VALUE_LEN]]
                                        prop_value += temp_value
                                        prop_mask += [vocab["<mask>"]] * \
                                            len(temp_value)
                                        prop_src += l["qualifiers"][qual_key]\
                                            .split()[:MAX_VALUE_LEN]

                                prop_value += [vocab["<eoc>"]]
                                prop_mask += [vocab["<eoc>"]]
                                prop_src += ["<eoc>"]
                                if len(datalist) + len(prop_value) > 350 \
                                        or datatype_count > 300:
                                    continue
                                datamask += prop_mask
                                datalist += prop_value
                                datapos += list(range(len(prop_value)))
                                datatype += [datatype_count] * len(prop_value)
                                datasrc += prop_src
                                datatype_count += 1
                        max_datatype_count = \
                            max(max_datatype_count, datatype_count)
                        assert len(datalist) == len(datamask), \
                            "{} != {}".format(len(datalist), len(datamask))
                        assert len(datalist) == len(datasrc), \
                            "{} != {}".format(len(datalist), len(datasrc))

                        if datalist:
                            wikidata[datajson["wikidata_name"]] = \
                                (datalist, datapos, datatype,
                                 datamask, datasrc)
                        else:
                            n_skip += 1
        self.expe.log.info("loaded #wikidata entries: {}, "
                           "found #duplicate entries: {}, "
                           "truncated max #data: {}, "
                           "non-truncated max #data: {}, "
                           "max #qual: {}, max #keys: {}, "
                           "max datatype count: {}, "
                           "#skip: {}"
                           .format(len(wikidata), dup, trunc_max_n_data,
                                   max_n_data, max_n_qual,
                                   max_n_keys, max_datatype_count - 1,
                                   n_skip))

        all_train_data = []
        n_train_has_wikidata = 0
        n_train_has_infobox = 0
        n_train_has_infobox_wikidata = 0

        max_datatype_count = 0
        max_datalist_len = 0
        max_st_len = 0
        max_dt_len = 0
        n_skip = 0
        n_break = 0
        for train_file in glob(train_path):
            with open(train_file) as fp:
                for nline, dataline in enumerate(fp):
                    if nline and nline % 100000 == 0:
                        self.expe.log.info(
                            "loading train file #line: {}".format(nline))
                    if dataline.strip():
                        datajson = json.loads(dataline.strip())

                        src_vocab = {UNK_WORD: 0}
                        wikidata_datalist = []
                        wikidata_datapos = []
                        wikidata_datatype = []
                        wikidata_datamask = []
                        wikidata_datasrc = []

                        infobox_datalist = []
                        infobox_datapos = []
                        infobox_datatype = []
                        infobox_datamask = []
                        infobox_datasrc = []

                        datatype_count = -1

                        if datajson["doc_title"] in wikidata:
                            # if wikidata_path is not None:
                            wikidata_datalist, wikidata_datapos, \
                                wikidata_datatype, wikidata_datamask, \
                                wikidata_datasrc = \
                                wikidata[datajson["doc_title"]]
                            datatype_count = wikidata_datatype[-1]

                            n_train_has_wikidata += 1

                        if datajson["doc_title"] in infobox:
                            infobox_datalist, infobox_datapos, \
                                infobox_datatype, infobox_datamask, \
                                infobox_datasrc = \
                                infobox[datajson["doc_title"]]

                            if datatype_count != -1:
                                n_train_has_infobox_wikidata += 1

                            datatype_count += 1
                            infobox_datatype = \
                                [idx + datatype_count
                                 for idx in infobox_datatype]
                            datatype_count = infobox_datatype[-1]
                            n_train_has_infobox += 1

                        wikidata_datasrc_ids = []
                        for w in wikidata_datasrc:
                            if w not in src_vocab:
                                src_vocab[w] = len(src_vocab)
                            wikidata_datasrc_ids.append(src_vocab[w])

                        infobox_datasrc_ids = []
                        for w in infobox_datasrc:
                            if w not in src_vocab:
                                src_vocab[w] = len(src_vocab)
                            infobox_datasrc_ids.append(src_vocab[w])

                        datatype_count += 1
                        doc_datalist = [vocab["<boc>"]]
                        doc_masklist = [vocab["<boc>"]]
                        if "<boc>" not in src_vocab:
                            src_vocab["<boc>"] = len(src_vocab)
                        doc_datasrc = [src_vocab["<boc>"]]

                        templist = []
                        for w in self.bpe.process_line("document title")\
                                .split()[:MAX_VALUE_LEN]:
                            if w not in src_vocab:
                                src_vocab[w] = len(src_vocab)
                            templist.append(vocab.get(w, 0))
                            doc_datasrc.append(src_vocab[w])

                        doc_datalist += templist
                        doc_masklist += [vocab["<mask>"]] * len(templist)

                        doc_datalist += [vocab["<bov>"]]
                        doc_masklist += [vocab["<bov>"]]
                        if "<bov>" not in src_vocab:
                            src_vocab["<bov>"] = len(src_vocab)
                        doc_datasrc += [src_vocab["<bov>"]]

                        templist = []
                        for w in datajson["doc_title_bpe"]\
                                .split()[:MAX_VALUE_LEN]:
                            if w not in src_vocab:
                                src_vocab[w] = len(src_vocab)
                            templist.append(vocab.get(w, 0))
                            doc_datasrc.append(src_vocab[w])
                        doc_datalist += templist
                        doc_masklist += [vocab["<mask>"]] * len(templist)

                        doc_datalist += [vocab["<eoc>"]]
                        doc_masklist += [vocab["<eoc>"]]
                        if "<eoc>" not in src_vocab:
                            src_vocab["<eoc>"] = len(src_vocab)
                        doc_datasrc += [src_vocab["<eoc>"]]
                        doc_datapos = list(range(len(doc_datalist)))
                        doc_datatype = [datatype_count] * len(doc_datalist)

                        datalist = []
                        masklist = []
                        datapos = []
                        datatype = []
                        datasrc = []

                        datatype_count += 1
                        for d in datajson["data"]:
                            curr_datalist = [vocab["<boc>"]]
                            curr_masklist = [vocab["<boc>"]]
                            curr_datasrc = [src_vocab["<boc>"]]

                            templist = []
                            for w in d[0].split()[:MAX_VALUE_LEN]:
                                if w not in src_vocab:
                                    src_vocab[w] = len(src_vocab)
                                templist.append(vocab.get(w, 0))
                                curr_datasrc.append(src_vocab[w])
                            curr_datalist += templist
                            curr_masklist += [vocab["<mask>"]] * len(templist)

                            curr_datalist += [vocab["<bov>"]]
                            curr_masklist += [vocab["<bov>"]]
                            curr_datasrc += [src_vocab["<bov>"]]

                            templist = []
                            for w in d[1].split()[:MAX_VALUE_LEN]:
                                if w not in src_vocab:
                                    src_vocab[w] = len(src_vocab)
                                templist.append(vocab.get(w, 0))
                                curr_datasrc.append(src_vocab[w])
                            curr_datalist += templist
                            curr_masklist += [vocab["<mask>"]] * len(templist)

                            curr_datalist += [vocab["<eoc>"]]
                            curr_masklist += [vocab["<eoc>"]]
                            if "<eoc>" not in src_vocab:
                                src_vocab["<eoc>"] = len(src_vocab)
                            curr_datasrc += [src_vocab["<eoc>"]]

                            if len(doc_datalist) + len(curr_datalist) + \
                                    len(datalist) + len(wikidata_datalist) + \
                                    len(infobox_datalist) > 500:
                                continue
                            if datatype_count + 1 >= 499:
                                continue
                            curr_datapos = list(range(len(curr_datalist)))
                            curr_datatype = \
                                [datatype_count] * len(curr_datalist)

                            datatype_count += 1

                            masklist += curr_masklist
                            datalist += curr_datalist
                            datapos += curr_datapos
                            datatype += curr_datatype
                            datasrc += curr_datasrc

                        section_datalist, section_datapos, \
                            section_datatype, section_masklist, \
                            section_datasrc = [], [], [], [], []
                        if datajson["sec_title"]:
                            section_masklist = [vocab["<boc>"]]
                            section_datalist = [vocab["<boc>"]]
                            section_datasrc = [src_vocab["<boc>"]]

                            templist = []
                            for w in self.bpe.process_line("section title")\
                                    .split()[:MAX_VALUE_LEN]:
                                if w not in src_vocab:
                                    src_vocab[w] = len(src_vocab)
                                templist.append(vocab.get(w, 0))
                                section_datasrc.append(src_vocab[w])
                            section_datalist += templist
                            section_masklist += [vocab["<mask>"]] * \
                                len(templist)

                            section_masklist += [vocab["<bov>"]]
                            section_datalist += [vocab["<bov>"]]
                            section_datasrc += [src_vocab["<bov>"]]

                            templist = []
                            for w in datajson["sec_title"][0]\
                                    .split()[:MAX_VALUE_LEN]:
                                if w not in src_vocab:
                                    src_vocab[w] = len(src_vocab)
                                templist.append(vocab.get(w, 0))
                                section_datasrc.append(src_vocab[w])

                            if "<subsec>" not in src_vocab:
                                src_vocab["<subsec>"] = len(src_vocab)
                            section_datalist += templist
                            section_masklist += [vocab["<mask>"]] * \
                                len(templist)
                            for s in datajson["sec_title"][1:]:

                                section_datalist += [vocab["<subsec>"]]
                                section_masklist += [vocab["<subsec>"]]
                                section_datasrc += [src_vocab["<subsec>"]]

                                templist = []
                                for w in s.split()[:MAX_VALUE_LEN]:
                                    if w not in src_vocab:
                                        src_vocab[w] = len(src_vocab)
                                    templist.append(vocab.get(w, 0))
                                    section_datasrc.append(src_vocab[w])
                                section_datalist += templist
                                section_masklist += [vocab["<mask>"]] * \
                                    len(templist)

                            section_datalist += [vocab["<eoc>"]]
                            section_masklist += [vocab["<eoc>"]]
                            section_datasrc += [src_vocab["<subsec>"]]
                            section_datapos = \
                                list(range(len(section_datalist)))
                            section_datatype = \
                                [datatype_count] * len(section_datalist)

                        if len(section_datalist) + len(doc_datalist) + \
                                len(datalist) + len(wikidata_datalist) + \
                                len(infobox_datalist) > 1000 \
                                or datatype_count >= 500:
                            n_skip += 1
                            continue
                        max_datatype_count = \
                            max(max_datatype_count, datatype_count)
                        max_datalist_len = \
                            max(max_datalist_len,
                                len(datalist) + len(wikidata_datalist) + len(doc_datalist)
                                )
                        max_st_len = max(max_st_len, len(section_datalist))
                        max_dt_len = max(max_dt_len, len(doc_datalist))
                        assert len(doc_datalist) == \
                            len(doc_masklist), "{} != {}".format(
                                len(doc_datalist), len(doc_masklist))
                        assert len(doc_datalist) == len(doc_datasrc), \
                            "{} != {}".format(
                                len(doc_datalist), len(doc_datasrc))
                        assert len(section_datalist) == len(section_masklist),\
                            "{} != {}".format(
                                len(section_datalist), len(section_masklist))
                        assert len(section_datalist) == len(section_datasrc), \
                            "{} != {}".format(
                                len(section_datalist), len(section_datasrc))
                        assert len(datalist) == len(masklist), \
                            "{} != {}".format(len(datalist), len(masklist))
                        assert len(datalist) == len(datasrc), \
                            "{} != {}".format(len(datalist), len(datasrc))
                        all_train_data.append(
                            {"idx": len(all_train_data),
                             "doc_title": (doc_datalist, doc_datapos,
                                           doc_datatype, doc_masklist,
                                           doc_datasrc),
                             "sec_title": (section_datalist,
                                           section_datapos, section_datatype,
                                           section_masklist, section_datasrc),
                             "wikidata": (wikidata_datalist, wikidata_datapos,
                                          wikidata_datatype, wikidata_datamask,
                                          wikidata_datasrc_ids),
                             "hyperlink_data": (datalist, datapos,
                                                datatype, masklist, datasrc),
                             "infobox": (infobox_datalist, infobox_datapos,
                                         infobox_datatype, infobox_datamask,
                                         infobox_datasrc_ids),
                             "src_vocab": src_vocab,
                             "inv_src_vocab": {i: str(w) for w, i
                                               in src_vocab.items()},
                             "text_src_vocab": [src_vocab.get(w, 0) for w
                                                in datajson["text"].split()[:self.expe.config.max_train_txt_len]],
                             "text": [vocab.get(w, 0) for w
                                      in datajson["text"].split()[:self.expe.config.max_train_txt_len]]
                             }
                        )

        self.expe.log.info(
            "loaded #train: {}, #has wikidata: {} ({:.2f}%), "
            "#has infobox: {} ({:.2f}%), #has infobox&wikidata: {} ({:.2f}%)"
            .format(len(all_train_data), n_train_has_wikidata,
                    n_train_has_wikidata / len(all_train_data) * 100
                    if len(all_train_data) else 0,
                    n_train_has_infobox,
                    n_train_has_infobox / len(all_train_data) * 100
                    if len(all_train_data) else 0,
                    n_train_has_infobox_wikidata,
                    n_train_has_infobox_wikidata / len(all_train_data) * 100
                    if len(all_train_data) else 0)
        )
        self.expe.log.info(
            "#skip: {}, max datatype count: {}, "
            "max datalist len: {}, "
            "max sec title len: {}, "
            "max doc title len: {}"
            .format(n_skip, max_datatype_count,
                    max_datalist_len, max_st_len, max_dt_len)
        )

        all_dev_data = []
        n_dev_has_wikidata = 0
        n_dev_has_infobox = 0
        n_dev_has_infobox_wikidata = 0

        max_datatype_count = 0
        max_datalist_len = 0
        max_st_len = 0
        max_dt_len = 0
        n_skip = 0
        for train_file in glob(dev_path):
            with open(train_file) as fp: #pylint: disable=C0103
                for nline, dataline in enumerate(fp):
                    if nline and nline % 100000 == 0:
                        self.expe.log.info("loading train file #line: {}".format(nline))
                    if dataline.strip():
                        datajson = json.loads(dataline.strip())

                        src_vocab = {UNK_WORD: 0}
                        wikidata_datalist = []
                        wikidata_datapos = []
                        wikidata_datatype = []
                        wikidata_datamask = []
                        wikidata_datasrc = []

                        infobox_datalist = []
                        infobox_datapos = []
                        infobox_datatype = []
                        infobox_datamask = []
                        infobox_datasrc = []

                        datatype_count = -1

                        if datajson["doc_title"] in wikidata:
                            # if wikidata_path is not None:
                            wikidata_datalist, wikidata_datapos, \
                                wikidata_datatype, wikidata_datamask, \
                                wikidata_datasrc = \
                                wikidata[datajson["doc_title"]]
                            datatype_count = wikidata_datatype[-1]

                            n_train_has_wikidata += 1

                        if datajson["doc_title"] in infobox:
                            infobox_datalist, infobox_datapos, \
                                infobox_datatype, infobox_datamask, \
                                infobox_datasrc = \
                                infobox[datajson["doc_title"]]

                            if datatype_count != -1:
                                n_train_has_infobox_wikidata += 1

                            datatype_count += 1
                            infobox_datatype = \
                                [idx + datatype_count for idx
                                 in infobox_datatype]
                            datatype_count = infobox_datatype[-1]
                            n_train_has_infobox += 1

                        wikidata_datasrc_ids = []
                        for w in wikidata_datasrc:
                            if w not in src_vocab:
                                src_vocab[w] = len(src_vocab)
                            wikidata_datasrc_ids.append(src_vocab[w])

                        infobox_datasrc_ids = []
                        for w in infobox_datasrc:
                            if w not in src_vocab:
                                src_vocab[w] = len(src_vocab)
                            infobox_datasrc_ids.append(src_vocab[w])

                        datatype_count += 1
                        doc_datalist = [vocab["<boc>"]]
                        doc_masklist = [vocab["<boc>"]]
                        if "<boc>" not in src_vocab:
                            src_vocab["<boc>"] = len(src_vocab)
                        doc_datasrc = [src_vocab["<boc>"]]

                        templist = []
                        for w in self.bpe.process_line("document title")\
                                .split()[:MAX_VALUE_LEN]:
                            if w not in src_vocab:
                                src_vocab[w] = len(src_vocab)
                            templist.append(vocab.get(w, 0))
                            doc_datasrc.append(src_vocab[w])

                        doc_datalist += templist
                        doc_masklist += [vocab["<mask>"]] * len(templist)

                        doc_datalist += [vocab["<bov>"]]
                        doc_masklist += [vocab["<bov>"]]
                        if "<bov>" not in src_vocab:
                            src_vocab["<bov>"] = len(src_vocab)
                        doc_datasrc += [src_vocab["<bov>"]]

                        templist = []
                        for w in datajson["doc_title_bpe"]\
                                .split()[:MAX_VALUE_LEN]:
                            if w not in src_vocab:
                                src_vocab[w] = len(src_vocab)
                            templist.append(vocab.get(w, 0))
                            doc_datasrc.append(src_vocab[w])
                        doc_datalist += templist
                        doc_masklist += [vocab["<mask>"]] * len(templist)

                        doc_datalist += [vocab["<eoc>"]]
                        doc_masklist += [vocab["<eoc>"]]
                        if "<eoc>" not in src_vocab:
                            src_vocab["<eoc>"] = len(src_vocab)
                        doc_datasrc += [src_vocab["<eoc>"]]
                        doc_datapos = list(range(len(doc_datalist)))
                        doc_datatype = [datatype_count] * len(doc_datalist)

                        datalist = []
                        masklist = []
                        datapos = []
                        datatype = []
                        datasrc = []

                        datatype_count += 1
                        for d in datajson["data"]:
                            curr_datalist = [vocab["<boc>"]]
                            curr_masklist = [vocab["<boc>"]]
                            curr_datasrc = [src_vocab["<boc>"]]

                            templist = []
                            for w in d[0].split()[:MAX_VALUE_LEN]:
                                if w not in src_vocab:
                                    src_vocab[w] = len(src_vocab)
                                templist.append(vocab.get(w, 0))
                                curr_datasrc.append(src_vocab[w])
                            curr_datalist += templist
                            curr_masklist += [vocab["<mask>"]] * len(templist)

                            curr_datalist += [vocab["<bov>"]]
                            curr_masklist += [vocab["<bov>"]]
                            curr_datasrc += [src_vocab["<bov>"]]

                            templist = []
                            for w in d[1].split()[:MAX_VALUE_LEN]:
                                if w not in src_vocab:
                                    src_vocab[w] = len(src_vocab)
                                templist.append(vocab.get(w, 0))
                                curr_datasrc.append(src_vocab[w])
                            curr_datalist += templist
                            curr_masklist += [vocab["<mask>"]] * len(templist)

                            curr_datalist += [vocab["<eoc>"]]
                            curr_masklist += [vocab["<eoc>"]]
                            if "<eoc>" not in src_vocab:
                                src_vocab["<eoc>"] = len(src_vocab)
                            curr_datasrc += [src_vocab["<eoc>"]]

                            if len(doc_datalist) + len(curr_datalist) + \
                                    len(datalist) + len(wikidata_datalist) + \
                                    len(infobox_datalist) > 500:
                                continue
                            if datatype_count + 1 >= 499:
                                continue
                            curr_datapos = list(range(len(curr_datalist)))
                            curr_datatype = \
                                [datatype_count] * len(curr_datalist)

                            datatype_count += 1

                            masklist += curr_masklist
                            datalist += curr_datalist
                            datapos += curr_datapos
                            datatype += curr_datatype
                            datasrc += curr_datasrc

                        section_datalist, section_datapos, section_datatype, \
                            section_masklist, section_datasrc = \
                            [], [], [], [], []
                        if datajson["sec_title"]:
                            section_masklist = [vocab["<boc>"]]
                            section_datalist = [vocab["<boc>"]]
                            section_datasrc = [src_vocab["<boc>"]]

                            templist = []
                            for w in self.bpe.process_line("section title")\
                                    .split()[:MAX_VALUE_LEN]:
                                if w not in src_vocab:
                                    src_vocab[w] = len(src_vocab)
                                templist.append(vocab.get(w, 0))
                                section_datasrc.append(src_vocab[w])
                            section_datalist += templist
                            section_masklist += \
                                [vocab["<mask>"]] * len(templist)

                            section_masklist += [vocab["<bov>"]]
                            section_datalist += [vocab["<bov>"]]
                            section_datasrc += [src_vocab["<bov>"]]

                            templist = []
                            for w in datajson["sec_title"][0]\
                                    .split()[:MAX_VALUE_LEN]:
                                if w not in src_vocab:
                                    src_vocab[w] = len(src_vocab)
                                templist.append(vocab.get(w, 0))
                                section_datasrc.append(src_vocab[w])

                            if "<subsec>" not in src_vocab:
                                src_vocab["<subsec>"] = len(src_vocab)
                            section_datalist += templist
                            section_masklist += \
                                [vocab["<mask>"]] * len(templist)
                            for s in datajson["sec_title"][1:]:

                                section_datalist += [vocab["<subsec>"]]
                                section_masklist += [vocab["<subsec>"]]
                                section_datasrc += [src_vocab["<subsec>"]]

                                templist = []
                                for w in s.split()[:MAX_VALUE_LEN]:
                                    if w not in src_vocab:
                                        src_vocab[w] = len(src_vocab)
                                    templist.append(vocab.get(w, 0))
                                    section_datasrc.append(src_vocab[w])
                                section_datalist += templist
                                section_masklist += \
                                    [vocab["<mask>"]] * len(templist)

                            section_datalist += [vocab["<eoc>"]]
                            section_masklist += [vocab["<eoc>"]]
                            section_datasrc += [src_vocab["<subsec>"]]
                            section_datapos = \
                                list(range(len(section_datalist)))
                            section_datatype = \
                                [datatype_count] * len(section_datalist)

                        max_datatype_count = \
                            max(max_datatype_count, datatype_count)
                        max_datalist_len = \
                            max(max_datalist_len,
                                len(datalist) + len(wikidata_datalist) + len(doc_datalist)
                                )
                        max_st_len = max(max_st_len, len(section_datalist))
                        max_dt_len = max(max_dt_len, len(doc_datalist))
                        assert len(doc_datalist) == len(doc_masklist), \
                            "{} != {}".format(
                                len(doc_datalist), len(doc_masklist))
                        assert len(doc_datalist) == len(doc_datasrc), \
                            "{} != {}".format(
                                len(doc_datalist), len(doc_datasrc))
                        assert len(section_datalist) == len(section_masklist),\
                            "{} != {}".format(
                                len(section_datalist), len(section_masklist))
                        assert len(section_datalist) == len(section_datasrc), \
                            "{} != {}".format(
                                len(section_datalist), len(section_datasrc))
                        assert len(datalist) == len(masklist), \
                            "{} != {}".format(len(datalist), len(masklist))
                        assert len(datalist) == len(datasrc), \
                            "{} != {}".format(len(datalist), len(datasrc))
                        all_dev_data.append(
                            {"idx": len(all_dev_data),
                             "doc_title": (doc_datalist, doc_datapos,
                                           doc_datatype, doc_masklist,
                                           doc_datasrc),
                             "sec_title": (section_datalist, section_datapos,
                                           section_datatype, section_masklist,
                                           section_datasrc),
                             "wikidata": (wikidata_datalist, wikidata_datapos,
                                          wikidata_datatype, wikidata_datamask,
                                          wikidata_datasrc_ids),
                             "hyperlink_data": (datalist, datapos, datatype,
                                                masklist, datasrc),
                             "infobox": (infobox_datalist, infobox_datapos,
                                         infobox_datatype, infobox_datamask,
                                         infobox_datasrc_ids),
                             "src_vocab": src_vocab,
                             "inv_src_vocab": {i: str(w) for w, i
                                               in src_vocab.items()},
                             "text_src_vocab": [src_vocab.get(w, 0) for w
                                                in datajson["text"].split()[:self.expe.config.max_train_txt_len]],
                             "text": [vocab.get(w, 0) for w
                                      in datajson["text"].split()[:self.expe.config.max_train_txt_len]],
                             "tok_text": " ".join(
                                datajson.get("tokenized_text", "")),
                             "untok_text": datajson["text"].replace("@@ ", "")
                             }
                        )

        self.expe.log.info(
            "loaded #dev: {}, #has wikidata: {} ({:.2f}%), "
            "#has infobox: {} ({:.2f}%), #has infobox&wikidata: {} ({:.2f}%)"
            .format(len(all_dev_data), n_dev_has_wikidata,
                    n_dev_has_wikidata / len(all_dev_data) * 100
                    if len(all_dev_data) else 0,
                    n_dev_has_infobox,
                    n_dev_has_infobox / len(all_dev_data) * 100
                    if len(all_dev_data) else 0,
                    n_dev_has_infobox_wikidata,
                    n_dev_has_infobox_wikidata / len(all_dev_data) * 100
                    if len(all_dev_data) else 0)
        )
        self.expe.log.info(
            "#skip: {}, max datatype count: {}, "
            "max datalist len: {}, "
            "max sec title len: {}, "
            "max doc title len: {}"
            .format(n_skip, max_datatype_count,
                    max_datalist_len, max_st_len, max_dt_len)
        )

        all_test_data = []
        self.expe.log.info("loaded #test: {}".format(len(all_test_data)))
        del wikidata
        del infobox
        return all_train_data, all_dev_data, all_test_data


class Minibatcher:
    @auto_init_args
    def __init__(self, data, save_dir, log, verbose, vocab_size,
                 filename, is_eval, batch_size, vocab,
                 return_wikidata, return_hyperlink,
                 input_wikidata, input_hyperlink, *args, **kwargs):
        self._reset()
        self.load(filename)

    def __len__(self):
        return len(self.idx_pool) - self.init_pointer

    def save(self, filename="minibatcher.ckpt"):
        path = os.path.join(self.save_dir, filename)
        pickle.dump([self.pointer, self.idx_pool], open(path, "wb"))
        if self.verbose:
            self.log.info("minibatcher saved to: {}".format(path))

    def load(self, filename="minibatcher.ckpt"):
        if self.save_dir is not None:
            path = os.path.join(self.save_dir, filename)
        else:
            path = None
        if self.save_dir is not None and os.path.exists(path):
            self.init_pointer, self.idx_pool = pickle.load(open(path, "rb"))
            self.pointer = self.init_pointer
            if self.verbose:
                self.log.info("loaded minibatcher from {}, init pointer: {}"
                              .format(path, self.init_pointer))
        else:
            if self.verbose:
                self.log.info("no minibatcher found at {}".format(path))

    def _reset(self):
        self.pointer = 0
        self.init_pointer = 0
        idx_list = np.arange(len(self.data))
        if not self.is_eval:
            np.random.shuffle(idx_list)
        self.idx_pool = [idx_list[i: i + self.batch_size]
                         for i in range(0, len(self.data), self.batch_size)]

    def _pad(self, data):
        max_text_len = max([len(d["text"]) for d in data])
        max_src_vocab_size = max([len(d["src_vocab"]) for d in data])
        if self.input_wikidata and self.input_hyperlink:
            max_input_data_len = \
                max([len(d["sec_title"][0]) +
                     len(d["doc_title"][0]) +
                     len(d["hyperlink_data"][0]) +
                     len(d["wikidata"][0]) +
                     len(d["infobox"][0])
                     for d in data])
        elif self.input_wikidata:
            max_input_data_len = \
                max([len(d["sec_title"][0]) +
                     len(d["doc_title"][0]) +
                     len(d["wikidata"][0]) +
                     len(d["infobox"][0])
                     for d in data])
        elif self.input_hyperlink:
            max_input_data_len = \
                max([len(d["sec_title"][0]) +
                    len(d["doc_title"][0]) +
                    len(d["hyperlink_data"][0])
                    for d in data])

        if self.return_wikidata and self.return_hyperlink:
            max_output_data_len = \
                max([len(d["sec_title"][0]) +
                    len(d["doc_title"][0]) +
                    len(d["hyperlink_data"][0]) +
                    len(d["wikidata"][0]) +
                    len(d["infobox"][0])
                    for d in data])
        elif self.return_wikidata:
            max_output_data_len = \
                max([len(d["sec_title"][0]) +
                    len(d["doc_title"][0]) +
                    len(d["wikidata"][0]) +
                    len(d["infobox"][0])
                    for d in data])
        elif self.return_hyperlink:
            max_output_data_len = \
                max([len(d["sec_title"][0]) +
                    len(d["doc_title"][0]) +
                    len(d["hyperlink_data"][0])
                    for d in data])

        input_data = \
            np.zeros((len(data), max_input_data_len)).astype("float32")
        input_data_mask = \
            np.zeros((len(data), max_input_data_len)).astype("float32")
        input_data_pos = \
            np.zeros((len(data), max_input_data_len)).astype("float32")
        input_data_type = \
            np.zeros((len(data), max_input_data_len)).astype("float32")
        input_if_hyp = \
            np.zeros((len(data), max_input_data_len)).astype("float32")
        input_data_src_vocab = \
            np.zeros((len(data), max_input_data_len)).astype("float32")
        input_data_src_tgt_vocab_map = \
            np.full((len(data), max_src_vocab_size), -1).astype("float32")

        tgt_inp_data = \
            np.zeros((len(data), max_output_data_len)).astype("float32")
        tgt_inp_data_mask = \
            np.zeros((len(data), max_output_data_len)).astype("float32")
        tgt_inp_data_pos = \
            np.zeros((len(data), max_output_data_len)).astype("float32")
        tgt_inp_data_type = \
            np.zeros((len(data), max_output_data_len)).astype("float32")
        tgt_inp_data_if_hyp = \
            np.zeros((len(data), max_output_data_len)).astype("float32")

        tgt_out_data = \
            np.zeros((len(data), max_output_data_len)).astype("float32")
        tgt_out_data_mask = \
            np.zeros((len(data), max_output_data_len)).astype("float32")

        tgt_input = \
            np.zeros((len(data), max_text_len + 1)).astype("float32")

        tgt_label = \
            np.zeros((len(data), max_text_len + 1)).astype("float32")
        tgt_label_src_vocab = \
            np.zeros((len(data), max_text_len + 1)).astype("float32")
        tgt_mask = \
            np.zeros((len(data), max_text_len + 1)).astype("float32")

        def get_hyp_only(d, idx):
            return d["hyperlink_data"][idx] + \
                d["doc_title"][idx] + \
                d["sec_title"][idx]

        def get_wikidata_only(d, idx):
            return d["doc_title"][idx] + \
                d["sec_title"][idx] + \
                d["wikidata"][idx] + \
                d["infobox"][idx]

        def get_all(d, idx):
            return d["hyperlink_data"][idx] + \
                d["doc_title"][idx] + \
                d["sec_title"][idx] + \
                d["wikidata"][idx] + \
                d["infobox"][idx]

        if self.input_wikidata and self.input_hyperlink:
            get_inp_d = get_all
        elif self.input_wikidata:
            get_inp_d = get_wikidata_only
        elif self.input_hyperlink:
            get_inp_d = get_hyp_only
        else:
            raise ValueError(
                "input_wikidata and input_hyperlink cannot both be False!")

        if self.return_wikidata and self.return_hyperlink:
            get_ret_d = get_all
        elif self.return_wikidata:
            get_ret_d = get_wikidata_only
        elif self.return_hyperlink:
            get_ret_d = get_hyp_only
        else:
            raise ValueError(
                "return_wikidata and return_hyperlink cannot both be False!")

        for i, d in enumerate(data):
            input_data[i, :len(get_inp_d(d, 0))] = \
                np.asarray(list(get_inp_d(d, 0))).astype("float32")
            input_data_mask[i, :len(get_inp_d(d, 0))] = 1.
            input_data_pos[i, :len(get_inp_d(d, 1))] = \
                np.asarray(list(get_inp_d(d, 1))).astype("float32")
            input_data_type[i, :len(get_inp_d(d, 2))] = \
                np.asarray(list(get_inp_d(d, 2))).astype("float32")
            input_data_src_vocab[i, :len(get_inp_d(d, 4))] = \
                np.asarray(list(get_inp_d(d, 4))).astype("float32")
            if self.input_hyperlink:
                input_if_hyp[i, :len(d["hyperlink_data"][2])] = 1.0

            for word, ids in d["src_vocab"].items():
                input_data_src_tgt_vocab_map[i, ids] = self.vocab.get(word, 0)

            assert sum(input_data_src_tgt_vocab_map[i][len(d["src_vocab"]):]) == \
                -1 * len(input_data_src_tgt_vocab_map[i][len(d["src_vocab"]):]),\
                "{} != {}".format(
                    sum(input_data_src_tgt_vocab_map[i][len(d["src_vocab"]):]),
                    -1 * len(input_data_src_tgt_vocab_map[i][len(d["src_vocab"]):]))
            assert sum((input_data_src_tgt_vocab_map[i][:len(d["src_vocab"])] != -1)) == \
                len(input_data_src_tgt_vocab_map[i][:len(d["src_vocab"])]), \
                "{} != {}".format(
                    sum((input_data_src_tgt_vocab_map[i][:len(d["src_vocab"])] != -1)),
                    len(input_data_src_tgt_vocab_map[i][:len(d["src_vocab"])]))

            tgt_mask_data = np.asarray(list(get_ret_d(d, 3))).astype("float32")
            tgt_orig_data = np.asarray(list(get_ret_d(d, 0))).astype("float32")
            mask_idx = tgt_mask_data == MASK_IDX
            tgt_recon_data = np.where(mask_idx, tgt_orig_data, UNK_IDX)
            tgt_out_data[i, :len(get_ret_d(d, 3))] = tgt_recon_data
            tgt_out_data_mask[i, :len(get_ret_d(d, 3))] = mask_idx

            tgt_inp_data[i, :len(get_ret_d(d, 3))] = tgt_mask_data
            tgt_inp_data_mask[i, :len(get_ret_d(d, 0))] = 1.
            tgt_inp_data_pos[i, :len(get_ret_d(d, 1))] = \
                np.asarray(list(get_ret_d(d, 1))).astype("float32")
            tgt_inp_data_type[i, :len(get_ret_d(d, 2))] = \
                np.asarray(list(get_ret_d(d, 2))).astype("float32")

            if self.return_hyperlink:
                tgt_inp_data_if_hyp[i, :len(d["hyperlink_data"][2])] = 1.0

            tgt_input[i, :len(d["text"]) + 1] = \
                np.asarray([BOS_IDX] + list(d["text"])).astype("float32")

            tgt_label[i, :len(d["text"]) + 1] = \
                np.asarray(list(d["text"]) + [EOS_IDX]).astype("float32")
            tgt_label_src_vocab[i, :len(d["text_src_vocab"]) + 1] = \
                np.asarray(list(d["text_src_vocab"]) + [0]).astype("float32")
            tgt_mask[i, :len(d["text"]) + 1] = 1.

        return [input_data, input_data_mask, input_data_pos, input_data_type,
                input_if_hyp, input_data_src_vocab,
                input_data_src_tgt_vocab_map,
                tgt_inp_data, tgt_inp_data_mask, tgt_inp_data_pos,
                tgt_inp_data_type, tgt_inp_data_if_hyp,
                tgt_out_data, tgt_out_data_mask,
                tgt_input, tgt_label, tgt_mask, tgt_label_src_vocab,
                [d["idx"] for d in data]]

    def __iter__(self):
        return self

    def __next__(self):
        if self.pointer == len(self.idx_pool):
            self._reset()
            raise StopIteration()

        idx = self.idx_pool[self.pointer]
        data = self.data[idx]
        self.pointer += 1
        return self._pad(data)
