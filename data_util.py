import time
import os
import torch
import re
import matplotlib.pyplot as plt
import numpy as np

from torch.utils.data import Dataset

from concurrent.futures import ProcessPoolExecutor,ThreadPoolExecutor
from functools import partial
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def airPLS_torch_gpu(X, lam=20, order=1, wep=0.3, p=0.05, itermax=20):

    m, n = X.shape

    I = torch.eye(n, device='cuda')
    D = I
    for _ in range(order):
        D = torch.diff(D, dim=0)
    DD = lam * (D.T @ D)


    Z = torch.zeros_like(X, device='cuda')

    for i in range(m):

        w = torch.ones(n, dtype=torch.float32, device='cuda')
        x = X[i, :]

        for j in range(itermax):

            W = torch.diag(w)

            C = torch.linalg.cholesky(W + DD)
            z = torch.linalg.solve(C.T, torch.linalg.solve(C, w * x))

            d = x - z
            dssn = torch.abs(d[d < 0]).sum()
            if dssn < 0.001 * torch.abs(x).sum():
                break

            w[d >= 0] = 0
            wep_count = int(n * wep)
            w[:wep_count] = p
            w[-wep_count:] = p
            w[d < 0] = torch.exp(j * torch.abs(d[d < 0]) / dssn)


        Z[i, :] = z

    Z = torch.min(Z, X)
    Xc = X - Z
    return Xc,Z
class SpectralDataset(Dataset):
    def __init__(self, root_dir, spectra_left=35, spectra_right=450, pH=None, PSA=None, time=None, threshold=-1,baseline_correction=False):

        self.spectra_left = spectra_left
        self.spectra_right = spectra_right
        self.threshold = threshold
        self.pH = pH
        self.PSA = PSA
        self.time = time
        self.baseline_correction = baseline_correction

        files = self._collect_files(root_dir)


        with ThreadPoolExecutor() as executor:
            results = executor.map(partial(self._process_file), files)


        self.spectra = []
        self.labels = []
        self.filenames = []
        self.baselines = []
        self.original_spectra = []
        for spectrum, label, filename, original_spectrum, baseline in results:
            if spectrum is not None:
                self.spectra.append(spectrum)
                self.labels.append(label)
                self.filenames.append(filename)
                self.baselines.append(baseline)
                self.original_spectra.append(original_spectrum)

    def _collect_files(self, root_dir):

        files = []
        for root, _, filenames in os.walk(root_dir):
            for filename in filenames:
                if filename.endswith(".txt"):
                    fixed_filename = filename.replace("点", ".").replace("o", "0")
                    file_path = os.path.join(root, fixed_filename)
                    match = re.search(r'pH([\d.]+)-([\d.]+)uMPSA(?:-\d+)?-(\d+)min', fixed_filename)
                    if match:

                        file_pH = float(match.group(1))
                        file_PSA = float(match.group(2))
                        file_time = int(match.group(3))

                        if (self.pH is None or file_pH == self.pH) and \
                                (self.PSA is None or file_PSA == self.PSA) and \
                                (self.time is None or file_time == self.time):
                            files.append((file_path, file_pH, file_PSA, file_time,filename))
        return files

    def _process_file(self, file_info):

        file_path, file_pH, file_PSA, file_time, filename = file_info
        try:

            with open(file_path, "r") as f:
                spectrum = np.array([float(line.split()[1]) for line in f if line.strip()])


            spectrum_max = spectrum.max()
            if (spectrum / spectrum_max).min() < self.threshold:
                return None, None, filename, None, None


            if len(spectrum) == 1044:
                spectrum = spectrum[12:1043]


            spectrum = spectrum[self.spectra_left:self.spectra_right]
            spectrum = spectrum / spectrum_max

            original_spectrum = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)


            if self.baseline_correction:
                spectrum_reshaped = spectrum.reshape(1, -1)
                corrected_spectrum,baseline = airPLS_torch_gpu(spectrum_reshaped, lam=20, order=1, wep=0.3, p=0.05, itermax=20)
                spectrum = corrected_spectrum.squeeze()
                # spectrum=spectrum/spectrum.max()
                baseline = baseline.squeeze()
                baseline_tensor = torch.tensor(baseline, dtype=torch.float32).unsqueeze(0)
            else:
                baseline_tensor = torch.zeros_like(original_spectrum)

            spectrum_tensor = torch.tensor(spectrum, dtype=torch.float32).unsqueeze(0)
            return spectrum_tensor, (file_pH, file_PSA, file_time), filename, original_spectrum, baseline_tensor
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None, None, filename, None, None

    def __len__(self):
        return len(self.spectra)

    def __getitem__(self, idx):
        return self.spectra[idx], self.labels[idx], self.filenames[idx], self.original_spectra[idx], self.baselines[idx]





