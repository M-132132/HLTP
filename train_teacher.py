from __future__ import print_function
import os
import re
import time
import torch
import torch.multiprocessing as mp
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import device
from utils import maskedMSE, maskedNLL, CELoss
from teacher_model.teacher_model import highwayNet
from loader2 import ngsimDataset  # 原 collate_fn

# ===========================
# 高效懒加载 PtDataset（多进程并行 + 文件加载进度）
# ===========================
class PtDataset(torch.utils.data.Dataset):
    def __init__(self, pt_dir):
        # 数字自然排序
        def extract_part_num(filename):
            match = re.search(r'part(\d+)\.pt$', filename)
            return int(match.group(1)) if match else float('inf')

        self.pt_files = sorted(
            [os.path.join(pt_dir, f) for f in os.listdir(pt_dir) if f.endswith('.pt')],
            key=extract_part_num
        )
        print(f"[INFO] Found {len(self.pt_files)} .pt files in {pt_dir}")

        # 扫描元数据（只读一次样本数，避免一次性加载全部数据）
        self.file_sizes = []
        for f in tqdm(self.pt_files, desc="[META] Scanning pt files"):
            try:
                part_data = torch.load(f, weights_only=False)
                self.file_sizes.append(len(part_data))
            except Exception as e:
                print(f"[ERROR] Read meta from {f} failed: {e}")
                self.file_sizes.append(0)

        # 样本索引映射 (全局样本id → 文件id, 文件内样本id)
        self.idx_map = []
        for file_idx, size in enumerate(self.file_sizes):
            for local_idx in range(size):
                self.idx_map.append((file_idx, local_idx))

        # 复用原 collate_fn
        self.collate_fn = ngsimDataset.collate_fn

        # 记录已加载过的文件（避免重复提示）
        self.loaded_files = set()

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        file_idx, local_idx = self.idx_map[idx]
        file_path = self.pt_files[file_idx]

        # 首次加载某个文件时提示
        if file_path not in self.loaded_files:
            print(f"[INFO] Loading {os.path.basename(file_path)} ...")
            self.loaded_files.add(file_path)

        part_data = torch.load(file_path, weights_only=False)
        return part_data[local_idx]

# ===========================
# 模型训练主函数
# ===========================
def train_main(trDataloader):
    args = {
        'use_cuda': True,
        'encoder_size': 64,
        'decoder_size': 128,
        'in_length': 30,
        'out_length': 25,
        'grid_size': (13, 3),
        'soc_conv_depth': 64,
        'conv_3x1_depth': 16,
        'dyn_embedding_size': 32,
        'input_embedding_size': 32,
        'num_lat_classes': 3,
        'num_lon_classes': 3,
        'use_maneuvers': True,
        'train_flag': True,
        'in_channels': 64,
        'out_channels': 64,
        'kernel_size': 3,
        'n_head': 4,
        'att_out': 48,
        'dropout': 0.2,
        'nbr_max': 39,
        'hidden_channels': 128
    }

    net = highwayNet(args)
    if args['use_cuda']:
        net = net.to(device)

    pretrainEpochs = 4
    trainEpochs = 12
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(pretrainEpochs + trainEpochs))
    lr = []

    for epoch_num in range(pretrainEpochs + trainEpochs):
        if epoch_num == 0:
            print('Pre-training with MSE loss')
        elif epoch_num == pretrainEpochs:
            print('Training with NLL loss')

        print("epoch:", epoch_num + 1, 'lr', optimizer.param_groups[0]['lr'])
        net.train_flag = True

        loss_gi1, loss_gix, loss_gx_2i, loss_gx_3i = 0, 0, 0, 0
        avg_tr_loss, avg_tr_time = 0, 0

        for i, data in enumerate(tqdm(trDataloader, desc=f"[TRAIN] Epoch {epoch_num+1}")):
            st_time = time.time()
            (
                hist_batch_stu, nbrs_batch_stu, lane_batch_stu, nbrslane_batch_stu, class_batch_stu,
                nbrsclass_batch_stu, va_batch_stu, nbrsva_batch_stu, fut_batch_stu,
                hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch,
                lane_batch, nbrslane_batch, class_batch, nbrsclass_batch,
                va_batch, nbrsva_batch, fut_batch, op_mask_batch, edge_index_batch,
                ve_matrix_batch, ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix
            ) = data

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

            fut_pred, lat_pred, lon_pred = net(
                hist_batch, nbrs_batch, mask_batch, lat_enc_batch, lon_enc_batch,
                lane_batch, nbrslane_batch, class_batch, nbrsclass_batch,
                va_batch, nbrsva_batch, edge_index_batch, ve_matrix_batch,
                ac_matrix_batch, man_matrix_batch, view_grip_batch, graph_matrix
            )

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
                print(f'mse: {loss_gi1/5000} | loss_gx_2: {loss_gx_2i/5000} | loss_gx_3: {loss_gx_3i/5000}')
                loss_gi1 = loss_gix = loss_gx_2i = loss_gx_3i = 0

        scheduler.step()
        torch.save(net.state_dict(), f'./checkpoints/model{epoch_num+1}.pth')

# ===========================
# Windows 主入口
# ===========================
if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)

    pt_dataset = PtDataset('../HLTP/train_pt_parts')
    print(f"[INFO] Total samples: {len(pt_dataset)}")

    trDataloader = DataLoader(
        pt_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,              # 根据CPU调整
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
        collate_fn=pt_dataset.collate_fn,
        pin_memory=True
    )

    train_main(trDataloader)
