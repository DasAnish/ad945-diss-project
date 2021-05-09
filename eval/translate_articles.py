import os
import pickle
from tqdm.notebook import tnrange
from src.transformer_layers import Transformer
from src.utils import translate_sentence
from src.sub_layers import PositionalEncoding
from src.text_preprocessing import create_models
from src.opt import Opt
import sacrebleu


def load_translator():
    """loading the chosen model and the sentence piece model"""
    opt = Opt.get_instance()

    model = Transformer(*opt.args)
    opt.save_model.load(f'{opt.path}/{opt.translator_model_file}')
    model.load_state_dict(opt.save_model.model_state_dict)
    model.encoder.positional_encoding = PositionalEncoding(opt.model_dim, 300).to(opt.device)
    model.decoder.positional_embeddings = PositionalEncoding(opt.model_dim, 300).to(opt.device)
    model.eval()
    create_models()
    return model


def translate(model):
    """For the model given, using beam-search to translate the news-artcies"""
    opt = Opt.get_instance()
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
                    translated = translate_sentence(sentence.lower(), model)
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

