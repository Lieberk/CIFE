import os
import sys
from easydict import EasyDict
import copy
from accelerate import Accelerator
from accelerate import DistributedDataParallelKwargs
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from models.CIFE import CIFE
from torch.utils.data import DataLoader
from tools.utils import *
from tools.log_utils import setting_logger
import yaml
from pprint import pformat
from tools.dataset import *
import warnings
from transformers import BartTokenizer
from tools.metrics_eval import eval_metrics

warnings.filterwarnings("ignore")


def _init_fn(worker_id):
    np.random.seed(2024)


def get_data(cfg):
    dataset_train = CIFE_Dataset(f'vid_train.txt', cfg)
    dataset_val = CIFE_Dataset(f'vid_val.txt', cfg)
    dataset_test = CIFE_Dataset(f'vid_test.txt', cfg)
    collate_fn = CIFE_collate_fn

    train_dataloader = DataLoader(dataset_train, batch_size=cfg.train.batch_size,
                                  num_workers=0,
                                  pin_memory=True,
                                  shuffle=True,
                                  worker_init_fn=_init_fn,
                                  collate_fn=collate_fn)
    val_dataloader = DataLoader(dataset_val, batch_size=cfg.eval.batch_size,
                                num_workers=0,
                                pin_memory=True,
                                shuffle=False,
                                worker_init_fn=_init_fn,
                                collate_fn=collate_fn)
    test_dataloader = DataLoader(dataset_test, batch_size=cfg.eval.batch_size,
                                 num_workers=0,
                                 pin_memory=True,
                                 shuffle=False,
                                 worker_init_fn=_init_fn,
                                 collate_fn=collate_fn)

    return train_dataloader, val_dataloader, test_dataloader


def gen_net(ep, model, loader, log_path, device):
    log_txt_name = os.path.join(log_path, f'gen_{ep}.txt')
    log_txt = open(log_txt_name, 'w', encoding="utf-8")
    model.eval()
    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            batch = send_to_device(batch, device)
            query_infer = model(**batch, mode='gen')
            log_txt.write('\n'.join(query_infer) + '\n')

    log_txt.flush()
    log_txt.close()
    gt_name = os.path.join(log_path, 'gt4test.txt')
    scores = eval_metrics(log_txt_name, gt_name)

    return scores


def eval_net(ep, model, loader, log_path, device, log):
    ppl_mean = 0
    model.eval()

    with torch.no_grad():
        for idx, batch in tqdm(enumerate(loader), total=len(loader)):
            batch = send_to_device(batch, device)
            ppl = model(**batch, mode='eval')
            ppl_mean += ppl.cpu().numpy()

    ppl_mean = ppl_mean / idx

    return ppl_mean


def run_stage(cfg, model, lr_sche, opt, train_loader, val_loader, test_loader, log_path, device, accelerator, log):
    print_every = int(len(train_loader) / 10)
    eval_every = 1

    max_epoch = cfg.train.max_epoch
    best_ppl = 1e6
    best_ppl_ep = 0

    best_model_wts_test = copy.deepcopy(model.state_dict())
    is_earlystop = False
    scores = []

    for epoch in range(max_epoch):
        if is_earlystop:
            break
        model.train()
        log.info(f"{'-' * 20} Current Epoch:  {epoch} {'-' * 20}")
        time_now = time.time()
        show_loss = 0

        for idx, batch in enumerate(train_loader):

            opt.zero_grad()
            batch_data = batch
            for k, v in batch_data.items():
                batch_data[k] = v.to(device)

            batch = send_to_device(batch, device)
            gen = model(**batch)
            loss_mean = sum([gen.loss])

            accelerator.backward(loss_mean)
            opt.step()

            cur_lr = opt.param_groups[-1]['lr']
            show_loss += loss_mean.detach().cpu().numpy()
            # print statistics
            if idx % print_every == print_every - 1 and accelerator.is_main_process:
                cost_time = time.time() - time_now
                time_now = time.time()
                log.info(
                    f'lr: {cur_lr:.6f} | step: {idx + 1}/{len(train_loader) + 1} '
                    f'| time cost {cost_time:.2f}s | loss: {(show_loss / print_every):.4f}')
                show_loss = 0

            lr_sche.step()

        if (epoch % eval_every) == (eval_every - 1) and epoch >= 0:
            log.info('Evaluating Net...')

            ppl = eval_net(epoch, model, val_loader, log_path, device, log)
            if ppl <= best_ppl:
                best_ppl = ppl
                best_ppl_ep = epoch
                best_model_wts_test = copy.deepcopy(model.state_dict())
                save_model(accelerator, model, log_path, epoch)
                log.info('Model Saved! ')
            else:
                if epoch - best_ppl_ep > cfg.train.epoch_stop - 1:
                    is_earlystop = True
                    print("early stopping...")

            log.info(f"Cur epoch: {epoch} | PPL: {ppl} | Best_ppl_ep: {best_ppl_ep} | Best ppl: {best_ppl}")

    model.load_state_dict(best_model_wts_test)
    log.info(f'Model Loaded! Best ep: {best_ppl_ep}')
    ppl = eval_net(best_ppl_ep, model, test_loader, log_path, device, log)
    score = gen_net(best_ppl_ep, model, test_loader, log_path, device)
    score['PPL'] = ppl
    print(score)
    scores.append(score)
    return score


def main(cfg):
    ddp = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[ddp])
    device = torch.device(cfg.GPU_id if torch.cuda.is_available() else 'cpu')

    setup_seed(int(cfg.train.seed))
    log_path = make_exp_dirs(cfg.name)
    log = setting_logger(log_path)
    tkr = BartTokenizer.from_pretrained(".\dataset\Pretrain/bart-base")

    model = CIFE(cfg, tkr)
    train_dataloader, val_dataloader, test_dataloader = get_data(cfg)

    optimizer = AdamW(model.parameters(), lr=cfg.train.pt_lr, weight_decay=1e-8)
    lr_scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=100,
                                                   num_training_steps=int(cfg.train.max_epoch) * len(train_dataloader))

    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    model = model.to(device)

    log.info(f'Found device: {device}')
    log.info(f"train data: {cfg.train.batch_size * len(train_dataloader)}")
    log.info(f"val data: {cfg.eval.batch_size * len(val_dataloader)}")
    log.info(f"test data: {cfg.eval.batch_size * len(test_dataloader)}")

    writr_gt(test_dataloader, log_path, tkr=tkr)

    scores = run_stage(cfg=cfg, model=model, lr_sche=lr_scheduler, opt=optimizer,
                       train_loader=train_dataloader, val_loader=val_dataloader, test_loader=test_dataloader,
                       log_path=log_path, device=device, accelerator=accelerator, log=log)
    log.info(pformat(scores))


if __name__ == '__main__':
    config_path = os.path.join('conf', 'basic_cfg.yaml')
    config = yaml.load(open(config_path, 'r'), Loader=yaml.Loader)
    config = EasyDict(config)
    main(config)
