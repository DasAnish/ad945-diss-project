from langdetect import detect
import os
from tqdm.notebook import tnrange
from src.opt import Opt
import pickle


def read_in_jsons():
    """Reads in the JSON from the opt.gv_crowd_path & gv_snippet_path"""
    opt = Opt.get_instance()
    with open(opt.gv_crowd_path, 'r') as f:
        opt.gv_crowd_json = f.read()
        opt.gv_crowd_json = eval(opt.gv_crowd_json)
    with open(opt.gv_snippet_path, 'r') as f:
        opt.gv_snippet_json = f.read()
        opt.gv_snippet_json = eval(opt.gv_snippet_json)


def filter_json(json_object):
    """Filter the jsons loaded in so that they have the required source language"""
    opt = Opt.get_instance()
    with open(json_object) as f:
        json_object = f.read()
    json_object = eval(json_object)
    filtered_json = []
    for item in json_object:
        if opt.src_lang in item['other_languages']:
            other_title = item['other_languages'][opt.src_lang]
            title = item['title']
            del item['other_languages']
            item['other_title'] = other_title
            item['src_file_path'] = f'{opt.eval_path}normalized/{opt.src_lang}/{other_title}.md'
            item['trg_file_path'] = f'{opt.eval_path}normalized/{opt.trg_lang}/{title}.md'
            filtered_json.append(item)
    return filtered_json


def join_jsons():
    """Join the JSON based on the items present in the crowd JSON"""
    opt = Opt.get_instance()
    temp = {}  # title: summary
    for item in opt.gv_snippet_json:
        temp[item['title']] = item['summary']
    filtered_json = []
    for item in opt.gv_crowd_json:
        if item['title'] in temp:
            item['snippet_summary'] = temp[item['title']]
            filtered_json.append(item)
    opt.eval_json = filtered_json


def filter_txt(l, txt, detect_lang=False):
    """Filtering the text to make sure that only reasonable sentences are present"""
    txt = [i for i in txt.split('\n') if i != '']
    if ' *[' in txt:
        txt = txt[:(txt.index(' *['))]

    new_txt = []
    for i in txt:
        try:
            lang = detect(i)
        except:
            continue
        else:
            if detect_lang:
                if lang == l:
                    new_txt.append(i)
            else:
                new_txt.append(i)
    txt = new_txt

    if 'Email' in txt:
        txt = txt[:(txt.index('Email')) - 2]

    return txt


def final_json_filter():
    """The final step is to read in the source and target sentences to the JSON"""
    opt = Opt.get_instance()
    temp_json = []
    tk0 = tnrange(len(opt.eval_json))
    for index in tk0:

        item = opt.eval_json[index]

        src_txt_file_path = f"{opt.src_txt_path}/{item['title']}"
        trg_txt_file_path = f"{opt.trg_txt_path}/{item['title']}"

        if os.path.exists(f"{src_txt_file_path}"):
            with open(src_txt_file_path) as f:
                src_txt = f.read().split('\n')
        else:
            with open(item['src_file_path']) as f:
                src_txt = f.read()
            src_txt = filter_txt(opt.src_lang, src_txt, detect_lang=True)

        if os.path.exists(trg_txt_file_path):
            with open(trg_txt_file_path) as f:
                trg_txt = f.read().split('\n')
        else:
            with open(item['trg_file_path']) as f:
                trg_txt = f.read()
            trg_txt = filter_txt(opt.trg_lang, trg_txt, detect_lang=True)

        m = 2
        if -m <= len(src_txt) - len(trg_txt) <= m:
            item['src_txt'] = src_txt
            item['trg_txt'] = trg_txt
            temp_json.append(item)
            # print(len(src_txt), len(trg_txt))

    opt.final_json = temp_json


def mkdir():
    """Making all the required directories/files to save intermediate results"""
    opt = Opt.get_instance()
    temp_path = f"{opt.eval_path}{opt.src_lang}"
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    opt.eval_src_path = temp_path

    temp_path = f"{opt.eval_path}{opt.src_lang}/{opt.translator_model_file}"
    if not os.path.exists(temp_path):
        os.mkdir(temp_path)
    opt.translated_text_path = temp_path

    # src_txt_path = f"{opt.eval_path}{opt.src_lang}/src_txt"
    if not os.path.exists(opt.src_txt_path):
        os.mkdir(opt.src_txt_path)

    if not os.path.exists(opt.trg_txt_path):
        os.mkdir(opt.trg_txt_path)

    if not os.path.exists(opt.summary_output_path):
        os.mkdir(opt.summary_output_path)


def preprocess_json():
    """Applying all of the smaller step detailed in the dissertaion in one function and saving the final preprocessed JSON as well"""
    opt = Opt.get_instance()
    if os.path.exists(opt.final_json_path):
        with open(opt.final_json_path, 'rb') as f:
            opt.final_json = pickle.load(f)
    else:
        opt.gv_snippet_json = filter_json(opt.gv_snippet_path)
        opt.gv_crowd_json = filter_json(opt.gv_crowd_path)
        join_jsons()
        final_json_filter()

        with open(opt.final_json_path, 'wb') as f:
            pickle.dump(opt.final_json, f)


def write_source_and_target():
    """Reading in and writing out the source and target sentences to the required location."""
    opt = Opt.get_instance()
    for item in opt.final_json:
        src_txt_file_path = f"{opt.src_txt_path}/{item['title']}"
        trg_txt_file_path = f"{opt.trg_txt_path}/{item['title']}"

        if not os.path.exists(src_txt_file_path):
            with open(src_txt_file_path, 'w') as f:
                f.write(' '.join(item['src_txt']))

        if not os.path.exists(trg_txt_file_path):
            with open(trg_txt_file_path, 'w') as f:
                f.write(' '.join(item['trg_txt']))
