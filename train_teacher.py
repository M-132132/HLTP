from __future__ import print_function
from tqdm import tqdm
import torch
from config import device
from utils import maskedMSE, maskedNLL, CELoss
from teacher_model.teacher_model import highwayNet
from loader2 import ngsimDataset
from torch.utils.data import DataLoader
import time


def main():
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

###模型初始化
    net = highwayNet(args)
    if args['use_cuda']:
        net = net.to(device)

###训练参数配置
    pretrainEpochs = 4
    trainEpochs = 12
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=(pretrainEpochs + trainEpochs))
    lr = []
    batch_size = 128

###数据加载器设置
    trSet = ngsimDataset('../HLTP/TrainSet.mat')
    trDataloader = DataLoader(trSet, batch_size=batch_size, shuffle=True, num_workers=8,
                              drop_last=True, persistent_workers=True, prefetch_factor=4,
                              collate_fn=trSet.collate_fn, pin_memory=True)

    # trDataloader = DataLoader(trSet, batch_size=batch_size, shuffle=True, num_workers=0,  # 改为0
    #                           drop_last=True, persistent_workers=False,  # 关闭persistent_workers
    #                           collate_fn=trSet.collate_fn, pin_memory=True)

###训练循环
    for epoch_num in range(pretrainEpochs + trainEpochs):
        if epoch_num == 0:
            print('Pre-training with MSE loss')
        elif epoch_num == pretrainEpochs:
            print('Training with NLL loss')
        print("epoch:", epoch_num + 1, 'lr', optimizer.param_groups[0]['lr'])
        net.train_flag = True   #设置模型为训练模式
        # 初始化各种损失和指标的累加器
        loss_gi1 = 0    # 轨迹预测损失累加器
        loss_gix = 0    # 总分类损失累加器
        loss_gx_2i = 0    # 纵向分类损失累加器
        loss_gx_3i = 0    # 横向分类损失累加器
        avg_tr_loss = 0    # 平均训练损失
        avg_tr_time = 0    # 平均训练时间
        avg_lat_acc = 0    # 横向分类准确率
        avg_lon_acc = 0    # 纵向分类准确率

        ###批次训练循环
        for i, data in enumerate(tqdm(trDataloader)):
            st_time = time.time()
            hist_batch_stu, nbrs_batch_stu, lane_batch_stu, nbrslane_batch_stu, class_batch_stu, nbrsclass_batch_stu, va_batch_stu, nbrsva_batch_stu, fut_batch_stu, hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch, lane_batch, nbrslane_batch, class_batch, nbrsclass_batch, va_batch, nbrsva_batch, \
                fut_batch, op_mask_batch, edge_index_batch, ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix = data

            ###数据移动到GPU
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

            ###前向传播
            fut_pred, lat_pred, lon_pred = net(hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch,
                                               lane_batch, nbrslane_batch, class_batch, nbrsclass_batch,
                                               va_batch, nbrsva_batch, edge_index_batch, ve_matrix_batch,
                                               ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix)

            ###计算损失
            if (epoch_num < pretrainEpochs) or (epoch_num >= (pretrainEpochs + trainEpochs) - 2):
                loss_g1 = maskedMSE(fut_pred, fut_batch, op_mask_batch)
            else:
                loss_g1 = maskedNLL(fut_pred, fut_batch, op_mask_batch)
            loss_gx_3 = CELoss(lat_pred, lat_enc_batch)    # 横向行为预测损失
            loss_gx_2 = CELoss(lon_pred, lon_enc_batch)    # 纵向行为预测损失
            loss_gx = loss_gx_3 + loss_gx_2    # 行为预测总损失
            loss_g = loss_g1 + 1 * loss_gx    # 最终总损失（轨迹预测损失 + 行为预测损失）

            ###反向传播和优化
            optimizer.zero_grad()
            loss_g.backward()
            a = torch.nn.utils.clip_grad_norm_(net.parameters(), 10)
            optimizer.step()


            ###记录训练指标
            batch_time = time.time() - st_time
            avg_tr_loss = avg_tr_loss + loss_g.item()
            avg_tr_time = avg_tr_time + batch_time
            loss_gi1 += loss_g1.item()
            loss_gx_2i += loss_gx_2.item()
            loss_gx_3i += loss_gx_3.item()
            loss_gix += loss_gx.item()
            lr.append(scheduler.get_lr()[0])


            if i % 5000 == 4999:
                print('mse:', loss_gi1 / 5000, '|loss_gx_2:', loss_gx_2i / 5000, '|loss_gx_3', loss_gx_3i / 5000)
                loss_gi1 = 0
                loss_gix = 0
                loss_gx_2i = 0
                loss_gx_3i = 0

        ###更新学习率和保存模型
        scheduler.step()
        epoch_num1 = epoch_num
        epoch_num1 = epoch_num1 + 1
        epoch_num1 = str(epoch_num1)
        torch.save(net.state_dict(), './checkpoints/model' + epoch_num1 + '.pth')


if __name__ == '__main__':
    main()
