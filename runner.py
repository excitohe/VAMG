import os
import ipdb
import matplotlib.pyplot as plt
import torch
import torch.utils.data as data

from loader import Dataset
from model import CMC_G, CMC_D, CMCLoss


def save_ckpt(state, filename):
    filename = f'{filename}.pth'
    torch.save(state, filename)


def load_ckpt(model, optimizer, filename, map_location, logger=None):
    if os.path.isfile(filename):
        logger.info(f'>>> Load ckpt from {filename}')
        checkpoint = torch.load(filename, map_location)
        epoch = checkpoint.get('epoch', -1)
        if model is not None and checkpoint['model_state'] is not None:
            model.load_state_dict(checkpoint['model_state'])
        if optimizer is not None and checkpoint['optimizer_state'] is not None:
            optimizer.load_state_dict(checkpoint['optimizer_state'])
        logger.info('>>> Load done !')
    else:
        raise FileNotFoundError
    return epoch


def model_state_to_cpu(model_state):
    model_state_cpu = type(model_state)()
    for key, val in model_state.items():
        model_state_cpu[key] = val.cpu()
    return model_state_cpu


def get_ckpt_state(G_model=None, D_model=None, G_optimizer=None, D_optimizer=None, epoch=None):
    G_optim_state = G_optimizer.state_dict() if G_optimizer is not None else None
    D_optim_state = D_optimizer.state_dict() if D_optimizer is not None else None
    if G_model is not None:
        if isinstance(G_model, torch.nn.DataParallel):
            G_model_state = model_state_to_cpu(G_model.module.state_dict())
        else:
            G_model_state = G_model.state_dict()
    else:
        G_model_state = None
    if D_model is not None:
        if isinstance(D_model, torch.nn.DataParallel):
            D_model_state = model_state_to_cpu(D_model.module.state_dict())
        else:
            D_model_state = D_model.state_dict()
    else:
        D_model_state = None
    ckpt_state = {
        'epoch': epoch,
        'G_model_state': G_model_state,
        'D_model_state': D_model_state,
        'G_optim_state': G_optim_state,
        'D_optim_state': D_optim_state,
    }
    return ckpt_state


def train(cfg, logger):
    device = torch.device('cuda')

    logger.info('>>> Build Modeling ...')
    cmc_g = CMC_G()
    cmc_d = CMC_D()
    cmc_g.to(device)
    cmc_d.to(device)

    compute_losses = CMCLoss(cfg)

    dataset = Dataset(cfg, is_train=True)
    dataloader = data.DataLoader(
        dataset=dataset,
        num_workers=8,
        batch_size=cfg.batch_size,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )

    G_optimizer = torch.optim.Adam(cmc_g.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))
    D_optimizer = torch.optim.Adam(cmc_d.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay, betas=(0.9, 0.99))

    total_iters = 0

    for epoch in range(cfg.epochs):
        epoch_iter = 0
        dataset.shuffle()
        disp_dict = {}

        for idx, batch_data in enumerate(dataloader):
            total_iters += cfg.batch_size
            epoch_iter += cfg.batch_size

            image, audio, unpair_image, unpair_audio, label, noise = batch_data
            image = image.to(device)
            audio = audio.to(device)
            unpair_image = unpair_image.to(device)
            unpair_audio = unpair_audio.to(device)
            label = label.to(device)
            noise = noise.to(device)

            data_dict = {
                'real_image': image,
                'real_audio': audio,
                'fake_image': unpair_image,
                'fake_audio': unpair_audio,
                'label': label,
                'noise': noise,
            }

            # Forward G
            G_stat_dict = cmc_g(image, audio, label, noise)

            # Backward D
            # compute_losses.set_requires_grad(model.D_CMC, True)
            D_optimizer.zero_grad()
            D_stat_dict = cmc_d(image, audio, unpair_image, unpair_audio, G_stat_dict, detach=True)
            D_loss_dict = compute_losses.optim_D(D_stat_dict)
            D_loss_sums = sum(loss for loss in D_loss_dict.values())
            D_loss_sums.backward()
            D_optimizer.step()

            # Backward G
            # compute_losses.set_requires_grad(model.D_CMC, False)
            G_optimizer.zero_grad()
            D_stat_dict = cmc_d(image, audio, unpair_image, unpair_audio, G_stat_dict, detach=False)
            G_loss_dict = compute_losses.optim_G(G_stat_dict, D_stat_dict, data_dict)
            G_loss_sums = sum(loss for loss in G_loss_dict.values())
            G_loss_sums.backward()
            G_optimizer.step()

            # # update D
            # compute_losses.set_requires_grad(model.D_CMC, True)
            # D_optimizer.zero_grad()
            # D_loss_dict = compute_losses.optim_D(stat_dict)
            # D_loss_sums = sum(loss for loss in D_loss_dict.values())
            # D_loss_sums.backward(retain_graph=True)
            # D_optimizer.step()

            # # repeat G
            # # for i in range(cfg.repeat_nums):
            # compute_losses.set_requires_grad(model.D_CMC, False)
            # G_optimizer.zero_grad()
            # G_loss_dict = compute_losses.optim_G(stat_dict)
            # G_loss_sums = sum(loss for loss in G_loss_dict.values())
            # G_loss_sums.backward()
            # G_optimizer.step()

            for key in D_loss_dict.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                disp_dict[key] += D_loss_dict[key]
            for key in G_loss_dict.keys():
                if key not in disp_dict.keys():
                    disp_dict[key] = 0
                disp_dict[key] += G_loss_dict[key]

            log_str = 'EPOCH[%03d/%03d] ITER[%05d]' % (epoch, cfg.epochs, epoch_iter)
            if total_iters % cfg.disp_freq == 0:
                for key in sorted(disp_dict.keys()):
                    disp_dict[key] = disp_dict[key] / cfg.disp_freq
                    log_str += ' %s:%.4f ' % (key, disp_dict[key])
                    disp_dict[key] = 0  # reset disp_dict
                logger.info(log_str)

        if epoch % cfg.save_freq == 0:
            os.makedirs(os.path.join(cfg.save_path, 'ckpt'), exist_ok=True)
            ckpt_name = os.path.join(cfg.save_path, 'ckpt', 'ckpt_epoch_%d' % epoch)
            save_ckpt(get_ckpt_state(cmc_g, cmc_d, G_optimizer, D_optimizer, epoch=epoch), ckpt_name)
            logger.info(f'>>> Save checkpoint: {ckpt_name}')
