import os
# print(os.getcwd())
# if os.getcwd() == r'D:/Desktop/Diss/ad945-diss-project/src':
#     os.chdir('..')
# print(os.getcwd())
import subprocess
from rouge import Rouge
import json
from mosestokenizer import MosesPunctuationNormalizer, MosesSentenceSplitter
import sentencepiece as spm

from TransformerLayers import Transformer

# translate
translate_source_file = ''
encoder_src_model_file = ''
encoder_trg_model_file = ''
encoder_src = spm.SentencePieceProcessor()
encoder_trg = spm.SentencePieceProcessor()
encoder_src.Init(encoder_src_model_file)
encoder_trg.Init(encoder_trg_model_file)

vocab_size = 8000
max_len = 200
path = 'models'
model_prefix = 'de-en-model-'
optim_file = 'data/optim_state_dict'
model_dim = 512
heads = 8
N = 6
args = (vocab_size, vocab_size, model_dim, model_dim*4, heads, N, max_len)

model = Transformer(*args)

with open(translate_source_file, 'r', encoding='utf-8') as f:
    src = f.read()
with MosesSentenceSplitter('en') as splitter:
    src_list = splitter([src])



# summarization
summary_input_file = 'data/summary_input.txt'
summary_interim_file = 'data/summary_interim.txt'
summary_output_file = 'data/summary_output.txt'
output = ''
# combining file's sentences into one
with open(summary_input_file, 'r', encoding='utf-8') as f:
    with MosesPunctuationNormalizer('en') as norm:
        for line in f.readlines():
            line = line.strip('\n').strip(' ')
            output += norm(line) + " "
with open(summary_interim_file, 'w', encoding='utf-8') as f:
    f.write(output)

summary_model = '../data/ada6_bridge_oldcopy_tagged_larger_acc_54.84_ppl_10.58_e17.pt'
# running summarizer
subprocess.run([
    'onmt_translate',
    '-beam_size', '10',
    '-model', summary_model,
    '-src', summary_interim_file,
    '-output', summary_output_file,
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

with open(summary_output_file, 'r', encoding='utf-8') as f:
    pred = f.read()

rouge = Rouge()

print(json.dumps(rouge.get_scores(pred, output)))
