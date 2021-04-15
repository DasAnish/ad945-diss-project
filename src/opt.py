import torch
from rouge import Rouge


class Opt:

    __instance = None

    @staticmethod
    def get_instance():
        if Opt.__instance is None:
            Opt()

        return Opt.__instance

    def __init__(self, src_lang='es'):

        if Opt.__instance is not None:
            raise Exception("This is a singleton class")
        else:
            Opt.__instance = self

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

        self.data_path = '/content/drive/MyDrive/Dissertation/data'

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
    def pc_input_file(self):
        return f'../data/{self.src_lang}/ParaCrawl.{self.src_lang}-{self.trg_lang}.'

    @property
    def nc_input_file(self):
        return f'../data/{self.src_lang}/News-Commentary.{self.src_lang}-{self.trg_lang}.'

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
        return f'{self.eval_path}gv/gv_crowd.json'

    @property
    def gv_snippet_path(self):
        return f'{self.eval_path}gv/gv_snippet.json'

    @property
    def test_restults_path(self):
        return f'{self.eval_path}test_results.json'

    @property
    def final_json_path(self):
        return f'{self.eval_path}final_json.{self.src_lang}.json'

    @property
    def path(self):
        return f'{self.data_path}/{self.src_lang}/{self.src_lang}-en-models'

    @property
    def model_prefix(self):
        return f'{self.src_lang}-en-model-'

    @property
    def translator_model_file(self):
        return self.model_prefix + str(self.model_num)

    @property
    def translated_path(self):
        return f'{self.data_path}/{self.src_lang}/model-{self.model_num}-translated'

    @property
    def summary_input_path(self):
        return f'{self.data_path}/{self.src_lang}/{self.translator_model_file}'

    @property
    def summary_output_path(self):
        return f'{self.data_path}/{self.src_lang}/summarized-{self.translator_model_file}'

    @property
    def summarized_path(self):
        return f'{self.data_path}/{self.src_lang}/model-{self.model_num}-summarized'

    @property
    def src_txt_path(self):
        return f"{self.data_path}/{self.src_lang}/src_txt"

    @property
    def trg_txt_path(self):
        return f"{self.data_path}/{self.src_lang}/trg_txt"

    @property
    def perf_trans_file_name(self):
        return f'{self.data_path}/{self.src_lang}/perfect-translation-summarized'

    @property
    def perf_trans_path(self):
        return f'{self.data_path}/{self.src_lang}/summarized-perfect-translation'


class Save:
    def __init__(self, model_state_dict=None, optim_state_dict=None):
        self.model_state_dict = model_state_dict
        self.optim_state_dict = optim_state_dict

    def save(self, filename):
        torch.save(self.model_state_dict, f'{filename}.model')
        torch.save(self.optim_state_dict, f'{filename}.optim')

    def load(self, filename):

        if torch.cuda.is_available():
            device = torch.device('cuda:0')
        else:
            device = torch.device('cpu')

        self.model_state_dict = torch.load(f'{filename}.model', map_location=device)
        self.optim_state_dict = torch.load(f'{filename}.optim', map_location=device)