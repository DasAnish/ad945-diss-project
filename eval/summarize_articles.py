import os
import subprocess
from tqdm.notebook import tnrange
from src.opt import Opt
import pickle


def perfect_trans_summary():
    opt = Opt.get_instance()
    if os.path.exists(opt.perf_trans_file_name):
        with open(opt.perf_trans_file_name, 'rb') as f:
            opt.perf_trans_summarized = pickle.load(f)
        return

    if not os.path.exists(opt.perf_trans_path):
        os.mkdir(opt.perf_trans_path)

    opt.perf_trans_summarized = []

    tk = tnrange(len(opt.final_json))
    for item in tk:
        item = opt.final_json[item]
        input_path = f'{opt.trg_txt_path}/{item["title"]}'
        output_path = f'{opt.perf_trans_path}/{item["title"]}'

        subprocess.run([
            'onmt_translate',
            '-beam_size', '10',
            '-model', opt.summary_model_path,
            '-src', input_path ,
            '-output', output_path,
            '-min_length', '35',
            '-stepwise_penalty',
            '-coverage_penalty', 'summary',
            '-length_penalty', 'wu',
            '-beta', '5',
            '-alpha', '0.9',
            '-block_ngram_repeat', '3'],
            stdout=subprocess.PIPE,
            encoding="utf-8"
        ).stdout.replace('‘', '\'').replace('’', '\'')

        with open(output_path, 'r') as f:
            pred = f.read()

        item = {'pred': pred}

        opt.perf_trans_summarized.append(item)

    with open(opt.perf_trans_file_name, 'wb') as f:
        pickle.dump(opt.perf_trans_summarized, f)


def summarize():
    opt = Opt.get_instance()
    for i in os.walk(opt.summary_input_path):
        break
    files = i[2]

    opt.summarized = []
    if os.path.exists(opt.summarized_path):
        with open(opt.summarized_path, 'rb') as f:
            opt.summarized = pickle.load(f)
        # print(i)
        return

    tk = tnrange(len(files))
    for i in tk:
        input_filename = files[i]
        output_filename = f'{opt.summary_output_path}/{input_filename}'
        input_filename = f'{opt.summary_input_path}/{input_filename}'
        source_filename = f''

        if os.path.exists(output_filename):
            with open(output_filename) as f:
                temp_txt = f.read()
            if len(temp_txt) > 0: continue

        subprocess.run([
            'onmt_translate',
            '-beam_size', '10',
            '-model', opt.summary_model_path,
            '-src', input_filename,
            '-output', output_filename,
            '-min_length', '35',
            '-stepwise_penalty',
            '-coverage_penalty', 'summary',
            '-length_penalty', 'wu',
            '-beta', '5',
            '-alpha', '0.9',
            '-block_ngram_repeat', '3'],
            stdout=subprocess.PIPE,
            encoding="utf-8"
        ).stdout.replace('‘', '\'').replace('’', '\'')

        with open(output_filename, 'r') as f:
            pred = f.read()

        item = {'pred': pred}

        opt.summarized.append(item)

    with open(opt.summarized_path, 'wb') as f:
        pickle.dump(opt.summarized, f)