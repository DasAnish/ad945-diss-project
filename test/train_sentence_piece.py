import sentencepiece as spm
import os
if os.getcwd() == r"D:\Desktop\Diss\ad945-diss-project\test":
    os.chdir("..")

# training the sentence piece model on the required input files
# also defining 3 new symbols for pad, start and end

spm.SentencePieceTrainer.Train(input='data/pairs.de',
                               model_prefix='data/pairs.de',
                               model_type='bpe',
                               vocab_size=2000,
                               user_defined_symbols=['<pad>', '<s>', '</s>'])

spm.SentencePieceTrainer.Train(input='data/pairs.en',
                               model_prefix='data/pairs.en',
                               model_type='bpe',
                               vocab_size=2000,
                               user_defined_symbols=['<pad>', '<s>', '</s>'])

