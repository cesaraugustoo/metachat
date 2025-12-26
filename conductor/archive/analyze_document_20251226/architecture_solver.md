# FiLM WaveY-Net Solver Analysis

## Overview
The `film-waveynet` module implements a semi-general fullwave surrogate solver using a Deep Learning approach. It leverages a UNet architecture augmented with Feature-wise Linear Modulation (FiLM) to conditionally model electromagnetic fields based on variable parameters like wavelength and source angle.

## Architecture (`source_code/multi_film_angle_dec_fwdadj_sample_learners.py`)

### Core Model: `UNet` with `FiLM`
- **Backbone:** A standard UNet (Encoder-Decoder) architecture responsible for processing the spatial structure of the metasurface.
- **Conditioning (FiLM):**
    - **Mechanism:** FiLM layers apply an affine transformation (`gamma * x + beta`) to the feature maps.
    - **Inputs:** `wavelength` and `angle`.
    - **Integration:** These conditions are passed through a linear layer (`nn.Linear(2, num_features * 2)`) to generate `gamma` and `beta` parameters, which then modulate the convolutional feature maps. This allows the single network to generalize across different operating conditions.

## Data Pipeline (`source_code/multi_film_angle_dec_fwdadj_sample_otf_dataloader.py`)

- **Dataset:** `SimulationDataset` loads simulation data.
- **Inputs:**
    - **Structure:** The refractive index distribution (`eps_r`) of the device.
    - **Source Fields:** The incident field distribution (`sources_`).
- **Targets:**
    - **Forward Fields:** The resulting magnetic field (`Hz_forward`).
    - **Adjoint Fields:** The fields from the adjoint simulation (`Hz_adjoint`), used for inverse design gradients.
- **Storage:** Data is stored in directories defined in `config.yaml`, with metadata in Parquet format.

## Training (`source_code/multi_film_angle_dec_fwdadj_sample_otf_train.py`)

- **Configuration:** Driven by `config.yaml` (hyperparameters, paths, device settings).
- **Optimization:**
    - Uses `torch.optim` (likely Adam).
    - Implements mixed-precision training (`torch.cuda.amp.GradScaler`, `autocast`) for efficiency.
- **Loss Function:**
    - While not explicitly read in the snippet, the import of `phys.py` and `consts.py` suggests a physics-informed loss or evaluation metric that respects Maxwell's equations.
- **Logging:** Uses `TensorBoard` (`SummaryWriter`) for tracking training progress and `matplotlib` for generating field visualizations during training.

## Physical Constraints (`source_code/phys.py`)
- This module likely contains the definitions for Maxwell's residual calculations, ensuring that the predicted fields satisfy the governing physical laws.

## Key Hyperparameters (from `config.yaml`)
- `architecture`: UNet
- `network_depth`: 6
- `num_kernels`: 16
- `batch_size`: 64
- `learning_rate`: 3e-4 to 1e-6 (schedule)
