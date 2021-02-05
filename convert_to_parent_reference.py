import json
import nltk
import sys

from glob import glob
from tqdm import tqdm

input_data_path = sys.argv[1]
infobox_path = sys.argv[2]
wikidata_path = sys.argv[3]
output_path = sys.argv[4]

dup = 0
trunc_max_n_data = 0
max_n_keys = 0
max_n_data = 0
max_n_qual = 0
max_datatype_count = 0
wikidata = {}
if wikidata_path is not None:
    with open(wikidata_path) as fp:
        for nline, dataline in tqdm(enumerate(fp)):
            if dataline.strip():
                datajson = json.loads(dataline.strip())
                datalist = ""
                datapos = []
                datatype = []
                datamask = []
                datatype_count = 0
                max_n_keys = max(max_n_keys, len(datajson["wikidata_details"]))
                for key in datajson["wikidata_details"]:
                    all_data_for_key = datajson["wikidata_details"][key]

                    for l in all_data_for_key:

                        temp_value = " ".join(
                            nltk.word_tokenize(key.replace("@@ ", "")))
                        prop_value = temp_value

                        temp_value = " ".join(
                            nltk.word_tokenize(l["data"].replace("@@ ", "")))
                        prop_value += "|||" + temp_value

                        if "qualifiers" in l:
                            max_n_qual = max(max_n_qual, len(l["qualifiers"]))
                            for qual_key in l["qualifiers"]:
                                temp_value = " ".join(
                                    nltk.word_tokenize(
                                        qual_key.replace("@@ ", "")))
                                prop_value += " " + temp_value
                                temp_value = " ".join(
                                    nltk.word_tokenize(
                                        l["qualifiers"][qual_key].replace("@@ ", "")))
                                prop_value += " " + temp_value

                        if prop_value.strip():
                            datalist += "\t" + " ".join(prop_value.split())

                wikidata[datajson["wikidata_name"]] = datalist.strip()
print("loaded #wikidata entries: {}".format(len(wikidata)))

infobox = {}
with open(infobox_path) as fp:
    for nline, dataline in enumerate(fp):
        if nline and nline % 50000 == 0:
            print("loading infobox #line: {}".format(nline))
        if dataline.strip():
            datajson = json.loads(dataline.strip())

            datalist = ""
            max_n_keys = max(max_n_keys, len(datajson["infobox"]))
            for key in datajson["infobox"]:
                all_data_for_key = datajson["infobox"][key]
                max_n_data = max(max_n_data, len(all_data_for_key))

                prop_value = " ".join(nltk.word_tokenize(key.replace("@@ ", "")))

                temp_value = " ".join(nltk.word_tokenize(all_data_for_key.replace("@@ ", "")))
                prop_value += "|||" + temp_value
                datalist += "\t" + prop_value

            if datalist:
                infobox[datajson["title"]] = datalist.strip()


all_train_data = []
n_train_has_wikidata = 0
max_datatype_count = 0
max_datalist_len = 0
max_st_len = 0
max_dt_len = 0
n_skip = 0
n_break = 0
fp_out = open(output_path, "w")
for train_file in glob(input_data_path):
    with open(train_file) as fp:
        for nline, dataline in enumerate(fp):
            if nline and nline % 100000 == 0:
                print("loading input file #line: {}".format(nline))
            if dataline.strip():
                datajson = json.loads(dataline.strip())

                wikidata_datalist = ""
                infobox_datalist = ""
                if datajson["doc_title"] in wikidata:
                    wikidata_datalist = wikidata[datajson["doc_title"]]
                    n_train_has_wikidata += 1
                if datajson["doc_title"] in infobox:
                    infobox_datalist = infobox[datajson["doc_title"]]
                datatype_count += 1

                templist = "document title"
                doc_datalist = templist

                templist = " ".join(nltk.word_tokenize(datajson["doc_title"]))
                doc_datalist += "|||" + templist

                datalist = ""
                masklist = []
                datapos = []
                datatype = []

                for d in datajson["data"]:
                    templist = " ".join(nltk.word_tokenize(d[0].replace("@@ ", "")))
                    curr_datalist = "\t" + templist

                    templist = " ".join(nltk.word_tokenize(d[1].replace("@@ ", "")))
                    curr_datalist += "|||" + templist

                    datalist += curr_datalist

                templist = "section title"
                section_datalist = templist

                templist = " ".join(nltk.word_tokenize(datajson["sec_title"][0].replace("@@ ", "")))
                section_datalist += "|||" + templist
                for s in datajson["sec_title"][1:]:

                    templist = " ".join(nltk.word_tokenize(s.replace("@@ ", "")))
                    section_datalist += "\t" + "section title" + "|||" + templist

                write_line = ""
                if doc_datalist.strip():
                    write_line += doc_datalist.strip()
                if section_datalist.strip():
                    write_line += "\t" + section_datalist.strip()
                if wikidata_datalist.strip():
                    write_line += "\t" + wikidata_datalist.strip()
                if datalist.strip():
                    write_line += "\t" + datalist.strip()
                if infobox_datalist.strip():
                    write_line += "\t" + infobox_datalist.strip()
                fp_out.write(write_line + "\n")
