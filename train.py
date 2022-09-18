import importlib
import torch
from torch import optim, nn
from torch.cuda import amp
from torch.distributed.optim import ZeroRedundancyOptimizer
from torchinfo import summary
import matplotlib.pyplot as plt
from random import uniform
from ignite.engine import Engine, Events
from ignite.handlers import Checkpoint, EMAHandler
from ignite.contrib.handlers.tensorboard_logger import *
from ignite.contrib.engines import common
from ignite import distributed as idist
import hydra
from omegaconf import DictConfig, OmegaConf
from itertools import chain
from contiguous_params import ContiguousParams

from utils import gamma2logas, get_instance, gamma2snr, snr2as, gamma2as
from loss import diffusion_elbo
from inference import reverse_process_new
from models import NoiseScheduler, LogSNRLinearScheduler


def get_logger(trainer, model, noise_scheduler, optimizer, log_dir, model_name):
    # Create a logger
    # start_time = datetime.now().strftime('%m%d_%H%M%S')
    tb_logger = common.setup_tb_logging(
        output_path=log_dir,
        trainer=trainer,
        optimizers=optimizer,
        log_every_iters=1
    )

    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsHistHandler(model)
    )
    tb_logger.attach(
        trainer,
        event_name=Events.EPOCH_COMPLETED,
        log_handler=WeightsHistHandler(noise_scheduler)
    )

    return tb_logger


def create_trainer(model: nn.Module,
                   noise_scheduler: nn.Module,
                   optimizer: ZeroRedundancyOptimizer,
                   scheduler: optim.lr_scheduler._LRScheduler,
                   device: torch.device,
                   cfg: DictConfig,
                   train_sampler,
                   checkpoint_path: str):
    rank = idist.get_rank()

    is_reduce_on_plateau = isinstance(
        scheduler, optim.lr_scheduler.ReduceLROnPlateau)

    scaler = amp.GradScaler(enabled=cfg.with_amp)

    if isinstance(noise_scheduler, nn.parallel.DistributedDataParallel) or isinstance(noise_scheduler, nn.parallel.DataParallel):
        base_noise_scheduler = noise_scheduler.module
    else:
        base_noise_scheduler = noise_scheduler

    def process_function(engine, batch):
        model.train()
        noise_scheduler.train()
        optimizer.zero_grad()

        if isinstance(batch, torch.Tensor):
            batch = (batch,)
        batch_on_device = [b.to(device) for b in batch]
        x, *c = batch_on_device
        noise = torch.randn_like(x)

        N = x.shape[0]
        # augmentation
        db_scale = 10 ** (torch.empty(N, device=device).uniform_(-6, 6) / 20)
        b_mask = torch.rand_like(db_scale) < 0.5
        db_scale[b_mask] = -db_scale[b_mask]
        x *= db_scale.unsqueeze(1)

        if cfg.train_T > 0:
            T = cfg.train_T
            s = torch.remainder(
                uniform(0, 1) + torch.arange(N, device=device) / N, 1.)
            s_idx = (s * T).long()
            t_idx = s_idx + 1

            t, s = t_idx / T, s_idx / T
            with amp.autocast(enabled=cfg.with_amp):
                gamma_ts, gamma_hat = noise_scheduler(torch.cat([t, s], dim=0))
                gamma_t, gamma_s = gamma_ts[:N], gamma_ts[N:]
                alpha_t, var_t = gamma2as(gamma_t)

                z_t = alpha_t[:, None] * x + var_t.sqrt()[:, None] * noise

                noise_hat = model(z_t, gamma_hat[:N], *c)

                loss, extra_dict = diffusion_elbo(
                    base_noise_scheduler.gamma0,
                    base_noise_scheduler.gamma1,
                    torch.expm1(gamma_t - gamma_s) * T,
                    x, noise, noise_hat)
        else:
            t = torch.remainder(
                uniform(0, 1) + torch.arange(N, device=device) / N, 1.)

            with amp.autocast(enabled=cfg.with_amp):
                gamma_t, gamma_hat = noise_scheduler(t)

                alpha_t, var_t = gamma2as(gamma_t)
                z_t = alpha_t[:, None] * x + var_t.sqrt()[:, None] * noise

                noise_hat = model(z_t, gamma_hat, *c)
                loss, extra_dict = diffusion_elbo(
                    base_noise_scheduler.gamma0,
                    base_noise_scheduler.gamma1,
                    base_noise_scheduler.gamma1 - base_noise_scheduler.gamma0,
                    x, noise, noise_hat)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        result = {'loss': loss.item()}
        result.update(extra_dict)
        return result

    trainer = Engine(process_function)

    to_save = {
        'model': model,
        'optimizer': optimizer,
        'scheduler': scheduler,
        'trainer': trainer,
        'noise_scheduler': noise_scheduler,
        'scaler': scaler
    }
    ema_model = None
    if rank == 0:
        ema_handler = EMAHandler(model, momentum=cfg.ema_momentum)
        ema_model = ema_handler.ema_model
        ema_handler.attach(trainer, name="ema_momentum",
                           event=Events.ITERATION_COMPLETED)
        to_save['ema_model'] = ema_model

    @trainer.on(Events.ITERATION_COMPLETED(every=10000))
    def consolidate_state_dict():
        optimizer.consolidate_state_dict()
        idist.barrier()

    common.setup_common_training_handlers(
        trainer,
        train_sampler=train_sampler,
        to_save=to_save if rank == 0 else None,
        save_every_iters=10000,
        output_path=cfg.save_dir,
        lr_scheduler=scheduler if not is_reduce_on_plateau else None,
        output_names=['loss'] +
        OmegaConf.to_container(cfg.extra_monitor_metrics),
        with_pbars=True if rank == 0 else False,
        with_pbar_on_iters=True,
        n_saved=2,
        log_every_iters=1,
        clear_cuda_cache=False
    )

    if is_reduce_on_plateau:
        trainer.add_event_handler(
            Events.ITERATION_COMPLETED, lambda engine: scheduler.step(
                engine.state.metrics['loss'])
        )

    if checkpoint_path:
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'ema_model' in to_save and 'ema_model' not in checkpoint:
            checkpoint['ema_model'] = checkpoint['model']
        Checkpoint.load_objects(
            to_load=to_save, checkpoint=checkpoint, strict=False)

    return trainer, ema_model


def get_dataflow(cfg: DictConfig):
    train_data = hydra.utils.instantiate(cfg.dataset)
    train_loader = idist.auto_dataloader(train_data, **cfg.loader)
    return train_loader


def initialize(cfg: DictConfig):
    model = hydra.utils.instantiate(cfg.model)
    model = idist.auto_model(model)

    if cfg.train_T > 0:
        noise_scheduler = NoiseScheduler()
    else:
        noise_scheduler = LogSNRLinearScheduler()
    noise_scheduler = idist.auto_model(noise_scheduler)

    parameters = ContiguousParams(
        chain(model.parameters(), noise_scheduler.parameters()))

    optim_kwargs: dict = OmegaConf.to_container(cfg.optimizer)
    *module_path, class_name = optim_kwargs.pop('_target_').split('.')
    m = importlib.import_module('.'.join(module_path))
    optim_type = getattr(m, class_name)
    optimizer = ZeroRedundancyOptimizer(
        parameters.contiguous(), optim_type, parameters_as_bucket_view=False, **optim_kwargs)
    # optimizer = idist.auto_optim(optimizer)

    scheduler = hydra.utils.instantiate(cfg.scheduler, optimizer=optimizer)

    return model, noise_scheduler, optimizer, scheduler


def training(local_rank, cfg: DictConfig):
    rank = idist.get_rank()
    device = idist.device()

    print(rank, ": run with config:", cfg, "- backend=", idist.backend())
    print(f'world size = {idist.get_world_size()}')

    *_, model_name = cfg.model._target_.split('.')
    checkpoint_path = cfg.checkpoint

    train_loader = get_dataflow(cfg)
    model, noise_scheduler, optimizer, scheduler = initialize(cfg)

    trainer, ema_model = create_trainer(model, noise_scheduler, optimizer,
                                        scheduler, device, cfg, train_loader.sampler,
                                        checkpoint_path)

    if rank == 0:
        # add model graph
        # use torchinfo
        for test_input in train_loader:
            break
        if isinstance(test_input, torch.Tensor):
            test_input = (test_input,)
        # test_input = test_input[:1].to(device)
        test_input_on_device = [t[:1].to(device) for t in test_input]
        x, *c = test_input_on_device
        t = torch.tensor([0.], device=device)
        summary(model.module if hasattr(model, 'module') else model,
                input_data=[x, t] + c,
                device=device,
                col_names=("input_size", "output_size", "num_params", "kernel_size",
                           "mult_adds"),
                col_width=16,
                row_settings=("depth", "var_names"))

        tb_logger = get_logger(
            trainer, model, noise_scheduler, optimizer, cfg.log_dir, model_name)

        @torch.no_grad()
        def generate_samples(engine):
            z_1 = torch.randn(1, cfg.sr * cfg.eval_dur, device=device)
            steps = torch.linspace(0, 1, cfg.eval_T, device=device)
            gamma, steps = noise_scheduler(steps)

            if cfg.speaker_emb_path:
                speaker_emb = torch.load(cfg.speaker_emb_path)
                speaker_emb = speaker_emb.to(device).requires_grad_(False)
                c = [speaker_emb.unsqueeze(0)]
            else:
                c = []

            z_0 = reverse_process_new(z_1, gamma,
                                      steps, ema_model, *c, with_amp=cfg.with_amp)
            generated = z_0.squeeze().clip(-0.99, 0.99)
            tb_logger.writer.add_audio(
                'generated', generated, engine.state.iteration, sample_rate=cfg.sr)

        @torch.no_grad()
        def plot_noise_curve(engine):
            figure = plt.figure()
            steps = torch.linspace(0, 1, 100, device=device)
            log_snr = -noise_scheduler(steps)[0].detach().cpu().numpy()
            steps = steps.cpu().numpy()
            plt.plot(steps, log_snr)
            tb_logger.writer.add_figure(
                'log_snr', figure, engine.state.iteration)

        trainer.add_event_handler(Events.EPOCH_COMPLETED, generate_samples)
        trainer.add_event_handler(Events.EPOCH_COMPLETED, plot_noise_curve)

    e = trainer.run(train_loader, max_epochs=cfg.epochs)

    if rank == 0:
        tb_logger.close()


@hydra.main(config_path="conf", config_name="config")
def run(cfg: DictConfig):
    backend = 'nccl'
    dist_configs = {
        'nproc_per_node': torch.cuda.device_count()
    }

    with idist.Parallel(backend=backend, **dist_configs) as parallel:
        parallel.run(training, cfg)


if __name__ == "__main__":
    run()
