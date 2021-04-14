import torch
from src.save_load import Save
from rouge import Rouge


class Opt:
    def __init__(self, src_lang='es'):
        self.src_lang = src_lang
        self.trg_lang = 'en'

        self.k = 10
        self.model_num = 1000 * 120

        self.num_mil = 1
        self.max_len = 150

        self.vocab_size = 8000
        self.tokensize = 4096
        self.print_every = 200
        self.save_every = 5000
        self.epochs = 10
        self.warmup_steps = 16000
        self.keep_training = False

        self.save_model = Save()

        self.eval_path = '/content/drive/MyDrive/Dissertation/eval/'
        self.summary_model_path = '/content/drive/MyDrive/Dissertation/summary_model/ada6_bridge_oldcopy_tagged_larger_acc_54.84_ppl_10.58_e17.pt'

        if torch.cuda.is_available():
            dev = 'cuda:0'
        else:
            dev = 'cpu'
        self.device = torch.device(dev)

        self.model_dim = 512
        self.heads = 8
        self.N = 6
        self.args = (self.vocab_size, self.vocab_size,
                     self.model_dim, self.model_dim * 4,
                     self.heads, self.N, self.max_len, 0.1, True)

        self.rouge = Rouge(metrics=['rouge-n', 'rouge-l', 'rouge-w'],
                          max_n = 2,
                          apply_avg = True,
                          weight_factor = 1.2,
                          length_limit=False)

    @property
    def input_file(self):
        return f'../data/{self.src_lang}/ParaCrawl.{self.src_lang}-{self.trg_lang}.'

    @property
    def model_file(self):
        return f'../data/{self.src_lang}/SPM-{self.num_mil}m-8k.{self.src_lang}-{self.trg_lang}.'

    @property
    def interim_file(self):
        return f'../data/{self.src_lang}/ParaCrawl.{self.src_lang}-{self.trg_lang}.{self.num_mil}m.'

    @property
    def dataset(self):
        return f'../data/{self.src_lang}/tokenized_dataset_{self.src_lang}_{self.num_mil}m'

    @property
    def dev_dataset(self):
        return f'../data/{self.src_lang}/DEV-{self.src_lang}-{self.trg_lang}.'

    @property
    def src_data_path(self):
        return self.interim_file + self.src_lang

    @property
    def trg_data_path(self):
        return self.interim_file + self.trg_lang

    @property
    def gv_crowd_path(self):
        return f'{self.eval_path}gv_crowd.json'

    @property
    def gv_snippet_path(self):
        return f'{self.eval_path}gv_snippet.json'

    @property
    def test_restults_path(self):
        return f'{self.eval_path}test_results.json'

    @property
    def final_json_path(self):
        return f'{self.eval_path}final_json.{self.src_lang}.json'

    @property
    def path(self):
        return f'/content/drive/MyDrive/Dissertation/{self.src_lang}-en-models'

    @property
    def model_prefix(self):
        return f'{self.src_lang}-en-model-'

    @property
    def translator_model_file(self):
        return self.model_prefix + str(self.model_num)

    @property
    def translated_path(self):
        return f'{self.eval_path}/{self.src_lang}/model-{self.model_num}-translated'

    @property
    def summary_input_path(self):
        return f'{self.eval_path}{self.src_lang}/{self.translator_model_file}'

    @property
    def summary_output_path(self):
        return f'{self.eval_path}{self.src_lang}/summarized-{self.translator_model_file}'

    @property
    def summarized_path(self):
        return f'{self.eval_path}{self.src_lang}/model-{self.model_num}-summarized'

    @property
    def src_txt_path(self):
        return f"{self.eval_path}{self.src_lang}/src_txt"

    @property
    def trg_txt_path(self):
        return f"{self.eval_path}{self.src_lang}/trg_txt"

    @property
    def perf_trans_file_name(self):
        return f'{self.eval_src_path}/perfect-translation-summarized'

    @property
    def perf_trans_path(self):
        return f'{self.eval_src_path}/summarized-perfect-translation'