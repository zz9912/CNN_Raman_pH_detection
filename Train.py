
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
import scipy.io as scio
from data_util import SpectralDataset,airPLS_torch_gpu
import os
from Model import ResNet
from tqdm import tqdm
import numpy as np
import random
from torchsummary import summary

seed = 1
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
generator = torch.Generator().manual_seed(seed)
np.set_printoptions(precision=4, suppress=True, threshold=np.inf)
np.set_printoptions(linewidth=200)
class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_score is None:
            self.best_score = val_loss
        elif val_loss > self.best_score - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_loss
            self.counter = 0


def split_dataset(dataset, train_ratio=0.7, val_ratio=0.1):
    train_size = int(train_ratio * len(dataset))
    val_size = int(val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    return random_split(dataset, [train_size, val_size, test_size],generator=generator)



def train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=200, model_save_path='2d-resnet.pt',base_corr=False):
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, verbose=True)
    early_stopping = EarlyStopping(patience=15, min_delta=1e-6)

    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:


            for spectra, labels, _, _, _ in train_loader:
                label1 = labels[0].clone().detach().to(torch.float32).unsqueeze(1)
                label1 = (label1 - 5) / 3
                label2 = labels[1].clone().detach().to(torch.float32).unsqueeze(1)
                spectra, label1, label2 = spectra.to(device), label1.to(device), label2.to(device)

                if base_corr:
                    spectra = spectra.squeeze(1)
                    corrected_spectrum, baseline = airPLS_torch_gpu(spectra, lam=20, order=1, wep=0.3, p=0.05,
                                                                    itermax=20)
                    corrected_spectrum = corrected_spectrum / corrected_spectrum.max(dim=1, keepdim=True).values
                    spectra = corrected_spectrum.clone().detach().unsqueeze(1)
                spectra = torch.abs(spectra.unsqueeze(-1) - spectra.unsqueeze(-2))

                optimizer.zero_grad()
                output1, output2 = model(spectra)
                loss1 = criterion(output1, label1)
                loss2 = criterion(output2, label2)

                loss = loss1 + loss2

                loss.backward()
                optimizer.step()
                train_loss += loss.item() * spectra.size(0)

                pbar.set_postfix({"Batch Loss": loss.item()})
                pbar.update(1)


        train_loss /= len(train_loader.dataset)


        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for spectra, labels, _, _, _ in val_loader:
                label1 = labels[0].clone().detach().to(torch.float32).unsqueeze(1)
                label1 = (label1 - 5) / 3
                label2 = labels[1].clone().detach().to(torch.float32).unsqueeze(1)

                spectra, label1, label2 = spectra.to(device), label1.to(device), label2.to(device)
                if base_corr:
                    spectra = spectra.squeeze(1)
                    corrected_spectrum, baseline = airPLS_torch_gpu(spectra, lam=20, order=1, wep=0.3, p=0.05,
                                                                    itermax=20)
                    corrected_spectrum = corrected_spectrum / corrected_spectrum.max(dim=1, keepdim=True).values
                    spectra = corrected_spectrum.clone().detach().unsqueeze(1)

                spectra = torch.abs(spectra.unsqueeze(-1) - spectra.unsqueeze(-2))

                output1, output2 = model(spectra)

                loss1 = criterion(output1, label1)
                loss2 = criterion(output2, label2)

                loss = loss1 + loss2

                val_loss += loss.item() * spectra.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(),model_save_path)
            print(f"New best model saved with Val Loss: {best_val_loss:.4f}")

        scheduler.step(val_loss)


        early_stopping(val_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break

def predict(model, test_loader,base_corr):
    model.eval()
    all_PH = []
    all_PSA = []
    all_label1 = []
    all_label2 = []
    all_names = []
    with torch.no_grad():
        for spectra, labels, name, _, _ in test_loader:
            label1 = labels[0].clone().detach().to(torch.float32).unsqueeze(1)
            label1 = (label1 - 5) / 3
            label2 = labels[1].clone().detach().to(torch.float32).unsqueeze(1)

            spectra, label1, label2 = spectra.to(device), label1.to(device), label2.to(device)
            if base_corr:
                spectra = spectra.squeeze(1)
                corrected_spectrum, baseline = airPLS_torch_gpu(spectra, lam=20, order=1, wep=0.3, p=0.05,
                                                                itermax=20)
                corrected_spectrum = corrected_spectrum / corrected_spectrum.max(dim=1, keepdim=True).values
                spectra = corrected_spectrum.clone().detach().unsqueeze(1)

            spectra = torch.abs(spectra.unsqueeze(-1) - spectra.unsqueeze(-2))

            output1, output2 = model(spectra) ## PH PSA
            all_PH.extend(output1.cpu().numpy().flatten())
            all_PSA.extend(output2.cpu().numpy().flatten())
            all_label1.extend(label1.cpu().numpy().flatten())
            all_label2.extend(label2.cpu().numpy().flatten())
            all_names.extend(name)
    return all_PH , all_PSA, all_label1, all_label2,all_names


if __name__ =='__main__':
    time_start = time.time()
    train_p=False
    predict_p=True
    base_corr = True
    show_result = True
    time_1=2
    data_root = "./dataset_demo/PSA_PH"
    model_path = "./checkpoint/2d_resnet_baseline_correction-demo.pt"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SpectralDataset(root_dir=data_root,spectra_left=35, spectra_right=450, pH=None, PSA=None, time=time_1,
                              threshold=-1,baseline_correction=False)

    train_dataset, val_dataset, test_dataset = split_dataset(dataset, train_ratio=0.7, val_ratio=0.1)

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    model = ResNet(layers=[3,4,6,3],num_classes1=1, num_classes2=1).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    #
    if train_p:
        train_model(model, criterion, optimizer, train_loader, val_loader, num_epochs=200, model_save_path=model_path,base_corr=base_corr)

    if predict_p:
        model.load_state_dict(torch.load(model_path, map_location=device))

        PH_pri, PSA_pri, label1, label2,name = predict(model, test_loader,base_corr=base_corr)

        predicted_PH = np.array([[3 * p + 5] for p in PH_pri])
        actual_PH = np.array([[3 * l + 5] for l in label1])
        com_PH = np.array([[3 * abs(a - b)] for a, b in zip(PH_pri, label1)])
        predicted_PSA = np.array([[p] for p in PSA_pri])
        actual_PSA = np.array([[l] for l in label2])
        com_PSA = np.array([[abs(a - b)] for a, b in zip(PSA_pri, label2)])
        name_list = np.array([[n] for n in name])

        # print(np.concatenate((predicted_PH, actual_PH, com_PH), axis=1))
        # print(np.concatenate((predicted_PSA, actual_PSA, com_PSA), axis=1))

        if show_result:
            actual_PH = np.round(actual_PH, 4)

            mae_pH = []
            sd_pH = []
            sse_pH = []
            mae_psa = []
            sd_psa = []
            sse_psa = []

            pH = [6, 6.25, 6.5, 6.75, 7, 7.25, 7.5]
            psa = [0.05, 0.1, 0.2, 0.5, 1]

            for pH_local in pH:
                index = np.where(actual_PH == pH_local)[0]
                pH_real = actual_PH[index]
                pH_pre = predicted_PH[index]

                mae_pH.append(np.mean(np.abs(pH_pre - pH_real)))
                sd_pH.append(np.std(np.abs(pH_pre - pH_real)))
                sse_pH.append(np.sum((pH_pre - pH_real) ** 2))

            mae_pH_total = np.mean(np.abs(actual_PH - predicted_PH))
            sd_pH_total = np.std(np.abs(actual_PH - predicted_PH))
            sse_pH_total = np.sum((actual_PH - predicted_PH) ** 2)

            for psa_local in psa:
                index = np.where(actual_PSA == psa_local)[0]
                psa_real = actual_PSA[index]
                psa_pre = predicted_PSA[index]

                mae_psa.append(np.mean(np.abs(psa_pre - psa_real)))
                sd_psa.append(np.std(np.abs(psa_pre - psa_real)))
                sse_psa.append(np.sum((psa_pre - psa_real) ** 2))

            mae_psa_total = np.mean(np.abs(actual_PSA - predicted_PSA))
            sd_psa_total = np.std(np.abs(actual_PSA - predicted_PSA))
            sse_psa_total = np.sum((actual_PSA - predicted_PSA) ** 2)

            print("pH MAE:", mae_pH)
            print("pH SD:", sd_pH)
            print("pH SSE:", sse_pH)
            print("Total pH MAE:", mae_pH_total)
            print("Total pH SD:", sd_pH_total)
            print("Total pH SSE:", sse_pH_total)

            print("PSA MAE:", mae_psa)
            print("PSA SD:", sd_psa)
            print("PSA SSE:", sse_psa)
            print("Total PSA MAE:", mae_psa_total)
            print("Total PSA SD:", sd_psa_total)
            print("Total PSA SSE:", sse_psa_total)

    time_end = time.time()
    time_usage = time_end - time_start
    print('timing', time_usage)

