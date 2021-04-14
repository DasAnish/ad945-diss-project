import torch
Tensor = torch.Tensor
from src.utils import *
from src.save_load import *
from src.text_preprocessing import *
import sacrebleu
import numpy as np
import time
from src.opt import Opt

Log().close()
log = Log()

opt = Opt()


create_fields(opt)
create_dataset(opt)
load_dev_dataset(opt)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"

device = torch.device(dev)
opt.device = device


def evaluate(model, opt):
    model.eval()
    start = time.time()

    refs = []
    hyp = []
    opt.skip = []

    tk2 = tnrange(len(opt.dev_src_sentences))
    for i in tk2:
        sentence = opt.dev_src_sentences[i]
        try:
            translated = translate_sentence(sentence.lower(), model, opt)
        except:
            opt.skip.append(i)
        else:
            refs.append(opt.dev_trg_sentences[i])
            hyp.append(translated)
            if i < 10:
                hyp_print = f"hyp: {hyp[-1]}"
                ref_print = f"ref: {refs[-1]}"
                opt.log.print(hyp_print, shell=False)
                opt.log.print(ref_print, shell=False)
                opt.log.print('', shell=False)
                opt.log.flush()

        if i == len(opt.dev_src_sentences) - 1:
            opt.hyp = hyp
            opt.refs = refs

            bleu = sacrebleu.corpus_bleu(opt.hyp[:-1], [opt.refs[:-1]])
            ter = sacrebleu.corpus_ter(opt.hyp[:-1], [opt.refs[:-1]])
            tk2.set_postfix_str(f"({ter} || {bleu})")
            log.print(f'{ter} || {bleu}', shell=False)

            return bleu, ter


def train_model(model, opt):
    model.train()

    start = time.time()

    total_loss = 0
    total_loss_at_save = 0
    last_loss = 0
    r = opt.save_every / opt.print_every
    epsilon = 0.1

    prev_scores = []
    tk0 = tnrange(opt.epochs)
    for epoch in tk0:
        temp = time.time()
        tk1 = tnrange(opt.train_len)
        batch_gen = batch(opt)
        last_batch_loss = 0
        for i in tk1:
            src_lis, trg_lis = next(batch_gen)
            opt.optim.zero_grad()
            try:
                src_tensor = torch.LongTensor(src_lis).to(opt.device)
                src_tensor.requires_grad = False
                trg_np = np.array(trg_lis)
                trg_tensor = torch.LongTensor(trg_np[:, :-1]).to(opt.device)
                trg_tensor.requires_grad = False
            except:
                del src_tensor
                continue
            src_mask, trg_mask = create_masks(src_tensor, trg_tensor, opt)

            preds = model(src_tensor, trg_tensor, src_mask, trg_mask)
            target = torch.LongTensor(trg_np[:, 1:]).to(opt.device).contiguous().view(-1)
            preds = preds.view(-1, preds.size(-1))
            loss = F.cross_entropy(preds, target, ignore_index=opt.trg_pad)

            loss.backward()
            total_loss += loss.item()
            optim.step()
            step = opt.starting_index + int(epoch * opt.train_len) + i + 1
            optim.param_groups[0]['lr'] = (opt.model_dim ** (-0.5)) * min(step ** (-0.5),
                                                                      step * (opt.warmup_steps ** (-1.5)))

            del src_mask, src_tensor, trg_mask, trg_tensor, preds, loss
            torch.cuda.empty_cache()

            if step % opt.print_every == 0:
                diff = total_loss - last_batch_loss
                diff = '%.3f' % diff
                last_batch_loss = total_loss

                avg = "%.3f" % (total_loss)
                t = "%.3f" % (time.time() - temp)
                tt = time.strftime('%H:%M:%S', time.gmtime(time.time() - start))

                output = f"time: {t}s, total: {tt}, loss = {avg}, step = {step}, diff = {diff}"
                log.print(output, shell=False)
                tk1.set_postfix_str(", " + output + '\n')

                total_loss_at_save += total_loss
                total_loss = 0
                temp = time.time()

            if step % opt.save_every == 0:
                model_name = f'{opt.path}/{opt.model_prefix}{step}'
                opt.save_model.model_state_dict = model.state_dict()
                opt.save_model.optim_state_dict = optim.state_dict()
                opt.save_model.save(model_name)

                avg = (total_loss_at_save / r)
                total_loss_at_save = 0
                diff = avg - last_loss
                last_loss = avg

                avg = '%.3f' % avg
                diff = '%.3f' % diff
                output = f"Saving model: {model_name} | avg_loss: {avg} | diff: {diff}"
                log.print(output, shell=False)
                tk0.set_postfix_str(',' + output + '\n')

            if step % opt.train_len == 0:
                bleu_score, ter = evaluate(model, opt)

                prev_scores.append(bleu_score.score)
                if len(prev_scores) > 5: prev_scores.pop(0)

                prev_avg_score = sum(prev_scores) / len(prev_scores)

                if len(prev_scores) == 3 and prev_avg_score - bleu_score.score < epsilon:
                    break

                log.print(f"{ter} || {bleu_score}")
                temp = time.time()
                model.train()

        log.flush()


if __name__ == '__main__':

    try:
        model, optim = load_model(opt)
        opt.optim = optim

        train_model(model, opt)

    except Exception as e:
        log.print(e, type=Log.ERROR)
        log.flush()
        raise e