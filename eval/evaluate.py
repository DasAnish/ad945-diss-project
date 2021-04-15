from eval.preprocessing import *
from eval.summarize_articles import perfect_trans_summary, summarize
import sacrebleu
from eval.translate_articles import *
import torch


def get_baseline_scores():
    opt = Opt.get_instance()
    read_in_jsons()
    mkdir()
    preprocess_json()
    write_source_and_target()
    perfect_trans_summary()

    for i in os.walk(opt.summary_input_path):
        break
    files = i[2]

    perfect_trans_preds = []
    first_50_preds = []
    snippets = []
    crowds = []

    tk = range(len(files))
    for i in tk:

        json = opt.final_json[i]

        snippets.append(json['snippet_summary'])
        crowds.append(json['summary'])

        _pred = opt.perf_trans_summarized[i]['pred']
        _pred = _pred.replace('<t>', '')
        _pred = _pred.replace('</t>', '')
        _pred = _pred.replace('.', '')
        _pred = _pred.replace('  ', ' ')
        perfect_trans_preds.append(_pred)

        trg_text_temp = ' '.join(json['trg_txt'])
        first_50_preds.append(' '.join(trg_text_temp.split(' ')[:50]))

    return {
            'rouge-snippet::first50': opt.rouge.get_scores(first_50_preds, snippets),
            'rouge-crowd::first50': opt.rouge.get_scores(first_50_preds, crowds),
            'rouge-snippet::perf-trans': opt.rouge.get_scores(perfect_trans_preds, snippets),
            'rouge-crowd::perf-trans': opt.rouge.get_scores(perfect_trans_preds, crowds)
            }


def get_scores():
    opt = Opt.get_instance()
    read_in_jsons()
    mkdir()
    preprocess_json()
    write_source_and_target()
    perfect_trans_summary()
    if opt.proper_method:
        model = load_translator()
        opt.k = 10
        translate(model)
        del model
        torch.cuda.empty_cache()
    else:
        translate(None)
    summarize()

    for i in os.walk(opt.summary_input_path):
        break
    files = i[2]

    preds = []
    perfect_trans_preds = []
    first_50_preds = []
    snippets = []
    crowds = []

    refs = []
    hyps = []

    tk = range(len(files))
    for i in tk:

        json = opt.final_json[i]

        pred = opt.summarized[i]['pred']
        if len(pred) == 0: continue

        json['pred_summary'] = pred
        pred = pred.replace('<t>', '')
        pred = pred.replace('</t>', '')
        pred = pred.replace('.', '')
        pred = pred.replace('  ', ' ')

        preds.append(pred)
        snippets.append(json['snippet_summary'])
        crowds.append(json['summary'])

        trg_text_temp = ' '.join(json['trg_txt'])
        first_50_preds.append(' '.join(trg_text_temp.split(' ')[:50]))

        # translation quality
        refs.append(trg_text_temp)
        hyps.append(' '.join(json['output']))

    return {'bleu': sacrebleu.corpus_bleu(hyps, [refs]),
            'rouge-perf-trans::trans-then-sum': opt.rouge.get_scores(preds, perfect_trans_preds),
            # 'rouge-snippet::trans-then-sum': rouge.get_scores(preds, snippets),#, avg=True),
            'rouge-crowd::trans-then-sum': opt.rouge.get_scores(preds, crowds)
            }
