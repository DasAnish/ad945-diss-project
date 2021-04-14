import os
import pickle
from tqdm.notebook import tnrange
from src.transformer_layers import Transformer
from src.utils import PositionalEncoding, translate_sentence
from src.text_preprocessing import create_fields
import sacrebleu


def load_translator(opt):
    model = Transformer(*opt.args)
    opt.save_model.load(f'{opt.path}/{opt.translator_model_file}')
    model.load_state_dict(opt.save_model.model_state_dict)
    model.encoder.positional_encoding = PositionalEncoding(opt.model_dim, 300).to(opt.device)
    model.decoder.positional_embeddings = PositionalEncoding(opt.model_dim, 300).to(opt.device)
    model.eval()
    create_fields(opt)
    return model


def translate(model, opt):
    opt.translated = []

    if os.path.exists(opt.translated_path):
        with open(opt.translated_path, 'rb') as f:
            opt.translated = pickle.load(f)

            for i, item in enumerate(opt.final_json):
                translated_item = opt.translated[i]
                item['output'] = translated_item['txt']
                item['bleu'] = translated_item['bleu']
                item['ter'] = translated_item['ter']

    else:
        tk1 = tnrange(len(opt.final_json))

        for i in tk1:
            translated_item = {}
            opt.translated.append(translated_item)

            item = opt.final_json[i]
            tk2 = tnrange(len(item['src_txt']), leave=False)
            output = []
            filename = f"{opt.eval_path}{opt.src_lang}/{opt.translator_model_file}/{item['title']}"
            trg_txt = item['trg_txt']

            for j in tk2:
                sentence = item['src_txt'][j]
                try:
                    translated = translate_sentence(sentence.lower(), model, opt)
                except Exception:
                    pass
                else:
                    output.append(translated)

            item['output'] = output

            hyp = [' '.join(output)]
            refs = [[' '.join(trg_txt)]]

            bleu = sacrebleu.corpus_bleu(hyp, refs)
            ter = sacrebleu.corpus_ter(hyp, refs)

            item['bleu'] = bleu
            item['ter'] = ter

            translated_item['bleu'] = bleu
            translated_item['ter'] = ter
            translated_item['txt'] = output

            tk1.set_postfix_str(str(bleu))

            with open(filename, 'w') as f:
                f.write(' '.join(output))
        with open(opt.translated_path, 'wb') as f:
            pickle.dump(opt.translated, f)











# import os
# print(os.getcwd())
# # if os.getcwd() == r'D:/Desktop/Diss/ad945-diss-project/src':
# #     os.chdir(r'D:/Desktop/Diss/ad945-diss-project/')
# # print(os.getcwd())
# import subprocess
# from rouge import Rouge
# import json
# from mosestokenizer import MosesPunctuationNormalizer, MosesSentenceSplitter
# import sentencepiece as spm
#
# from TransformerLayers import Transformer
#
# # translate
# translate_source_file = ''
# encoder_src_model_file = ''
# encoder_trg_model_file = ''
# encoder_src = spm.SentencePieceProcessor()
# encoder_trg = spm.SentencePieceProcessor()
# encoder_src.Init(encoder_src_model_file)
# encoder_trg.Init(encoder_trg_model_file)
#
# vocab_size = 8000
# max_len = 200
# path = 'models'
# model_prefix = 'de-en-model-'
# optim_file = 'data/optim_state_dict'
# model_dim = 512
# heads = 8
# N = 6
# args = (vocab_size, vocab_size, model_dim, model_dim*4, heads, N, max_len)
#
# model = Transformer(*args)
#
# # with open(translate_source_file, 'r', encoding='utf-8') as f:
# #     src = f.read()
# # with MosesSentenceSplitter('en') as splitter:
# #     src_list = splitter([src])
#
#
#
# # summarization
# summary_input_file = 'data/summary_input.txt'
# summary_interim_file = 'data/summary_interim.txt'
# summary_output_file = 'data/summary_output.txt'
# output = ''
# def norm(x):
#     return x
# # combining file's sentences into one
# with open(summary_input_file, 'r', encoding='utf-8') as f:
#     # with MosesPunctuationNormalizer('en') as norm:
#     for line in f.readlines():
#         line = line.strip('\n').strip(' ')
#         output += norm(line) + " "
# with open(summary_interim_file, 'w', encoding='utf-8') as f:
#     f.write(output)
#
# summary_model = '../data/ada6_bridge_oldcopy_tagged_larger_acc_54.84_ppl_10.58_e17.pt'
# # running summarizer
# subprocess.run([
#     'onmt_translate',
#     '-beam_size', '20',
#     '-model', summary_model,
#     '-src', summary_interim_file,
#     '-output', summary_output_file,
#     '-min_length', '50',
#     '-stepwise_penalty',
#     '-coverage_penalty', 'summary',
#     '-length_penalty', 'wu',
#     '-beta', '5',
#     '-alpha', '0.9',
#     '-block_ngram_repeat', '4',
#     '-threshold', '1.5'],
#     stdout=subprocess.PIPE,
#     encoding="utf-8"
# ).stdout.replace('‘', '\'').replace('’', '\'')
#
# with open(summary_output_file, 'r', encoding='utf-8') as f:
#     pred = f.read()
#
# rouge = Rouge()
#
# print(
#     (rouge.get_scores(pred, output)))
