from __future__ import print_function
from tqdm import tqdm
import torch
import torch.multiprocessing as mp
from config import device
from utils import maskedMSE, maskedNLL, CELoss
from teacher_model.teacher_model import highwayNet
from loader2 import ngsimDataset
from torch.utils.data import DataLoader
import time


def train_main():
    args = {}
    args['use_cuda'] = True
    args['encoder_size'] = 64
    args['decoder_size'] = 128
    args['in_length'] = 30
    args['out_length'] = 25
    args['grid_size'] = (13, 3)
    args['soc_conv_depth'] = 64
    args['conv_3x1_depth'] = 16
    args['dyn_embedding_size'] = 32
    args['input_embedding_size'] = 32
    args['num_lat_classes'] = 3
    args['num_lon_classes'] = 3
    args['use_maneuvers'] = True
    args['train_flag'] = True
    args['in_channels'] = 64
    args['out_channels'] = 64
    args['kernel_size'] = 3
    args['n_head'] = 4
    args['att_out'] = 48
    args['dropout'] = 0.2
    args['nbr_max'] = 39
    args['hidden_channels'] = 128

    ### 模型初始化
    net = highwayNet(args)
    if args['use_cuda']:
        net = net.to(device)

    ### 训练参数配置
    pretrainEpochs = 4
    trainEpochs = 12
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer=optimizer,
        T_max=(pretrainEpochs + trainEpochs)
    )
    lr = []
    batch_size = 128

    # ### 数据加载器设置（放在 main 里，保证 Windows 多进程安全）
    # trSet = ngsimDataset('../HLTP/TrainSet.mat')
    # trDataloader = DataLoader(
    #     trSet,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     num_workers=4,  # Windows 优化，建议 4 或 CPU 核心 - 1
    #     drop_last=True,
    #     persistent_workers=True,  # 多进程保持存活，加速下一个 epoch
    #     prefetch_factor=2,        # 不要太大，防止 Windows 卡死
    #     collate_fn=trSet.collate_fn,
    #     pin_memory=True
    # )

    ### 训练循环
    for epoch_num in range(pretrainEpochs + trainEpochs):
        if epoch_num == 0:
            print('Pre-training with MSE loss')
        elif epoch_num == pretrainEpochs:
            print('Training with NLL loss')

        print("epoch:", epoch_num + 1, 'lr', optimizer.param_groups[0]['lr'])
        net.train_flag = True

        loss_gi1, loss_gix, loss_gx_2i, loss_gx_3i = 0, 0, 0, 0
        avg_tr_loss, avg_tr_time = 0, 0

        for i, data in enumerate(tqdm(trDataloader)):
            st_time = time.time()
            (
                hist_batch_stu, nbrs_batch_stu, lane_batch_stu, nbrslane_batch_stu, class_batch_stu,
                nbrsclass_batch_stu, va_batch_stu, nbrsva_batch_stu, fut_batch_stu,
                hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch,
                lane_batch, nbrslane_batch, class_batch, nbrsclass_batch,
                va_batch, nbrsva_batch, fut_batch, op_mask_batch, edge_index_batch,
                ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix
            ) = data

            ### 数据移动到 GPU
            if args['use_cuda']:
                hist_batch = hist_batch.to(device)
                nbrs_batch = nbrs_batch.to(device)
                mask_batch = mask_batch.to(device)
                lat_enc_batch = lat_enc_batch.to(device)
                lon_enc_batch = lon_enc_batch.to(device)
                lane_batch = lane_batch.to(device)
                nbrslane_batch = nbrslane_batch.to(device)
                class_batch = class_batch.to(device)
                nbrsclass_batch = nbrsclass_batch.to(device)
                fut_batch = fut_batch.to(device)
                op_mask_batch = op_mask_batch.to(device)
                va_batch = va_batch.to(device)
                nbrsva_batch = nbrsva_batch.to(device)
                edge_index_batch = edge_index_batch.to(device)
                ve_matrix_batch = ve_matrix_batch.to(device)
                ac_matrix_batch = ac_matrix_batch.to(device)
                man_matrix_batch = man_matrix_batch.to(device)
                view_grip_batch = view_grip_batch.to(device)
                graph_matrix = graph_matrix.to(device)

            ### 前向传播
            fut_pred, lat_pred, lon_pred = net(
                hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch,
                lane_batch, nbrslane_batch, class_batch, nbrsclass_batch,
                va_batch, nbrsva_batch, edge_index_batch, ve_matrix_batch,
                ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix
            )

            ### 损失计算
            if (epoch_num < pretrainEpochs) or (epoch_num >= (pretrainEpochs + trainEpochs) - 2):
                loss_g1 = maskedMSE(fut_pred, fut_batch, op_mask_batch)
            else:
                loss_g1 = maskedNLL(fut_pred, fut_batch, op_mask_batch)

            loss_gx_3 = CELoss(lat_pred, lat_enc_batch)
            loss_gx_2 = CELoss(lon_pred, lon_enc_batch)
            loss_gx = loss_gx_3 + loss_gx_2
            loss_g = loss_g1 + loss_gx

            optimizer.zero_grad()
            loss_g.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()

            batch_time = time.time() - st_time
            avg_tr_loss += loss_g.item()
            avg_tr_time += batch_time
            loss_gi1 += loss_g1.item()
            loss_gx_2i += loss_gx_2.item()
            loss_gx_3i += loss_gx_3.item()
            loss_gix += loss_gx.item()
            lr.append(scheduler.get_last_lr()[0])

            if i % 5000 == 4999:
                print(f'mse: {loss_gi1 / 5000} | loss_gx_2: {loss_gx_2i / 5000} | loss_gx_3: {loss_gx_3i / 5000}')
                loss_gi1 = 0
                loss_gix = 0
                loss_gx_2i = 0
                loss_gx_3i = 0

        scheduler.step()
        torch.save(net.state_dict(), f'./checkpoints/model{epoch_num+1}.pth')


if __name__ == '__main__':
    # mp.set_start_method('spawn', force=True)  # Windows 多进程安全启动
    # train_main()

    import torch.multiprocessing as mp
    mp.set_start_method('spawn', force=True)

    trSet = ngsimDataset('../HLTP/TrainSet.mat')
    trDataloader = DataLoader(
        trSet,
        batch_size=128,
        shuffle=True,
        num_workers=12,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=3,
        collate_fn=trSet.collate_fn,
        pin_memory=True
    )

    train_main()
