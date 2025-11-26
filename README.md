# ⚡ FlashVSR

> **📌 Note:** This is a fork of [FlashVSR](https://github.com/OpenImagingLab/FlashVSR). The original repository is maintained by the OpenImagingLab team.

**Towards Real-Time Diffusion-Based Streaming Video Super-Resolution**

**Original Authors:** Junhao Zhuang, Shi Guo, Xin Cai, Xiaohui Li, Yihao Liu, Chun Yuan, Tianfan Xue

<img src="./examples/WanVSR/assets/teaser.png" />

---

### 🌟 Abstract

Diffusion models have recently advanced video restoration, but applying them to real-world video super-resolution (VSR) remains challenging due to high latency, prohibitive computation, and poor generalization to ultra-high resolutions. Our goal in this work is to make diffusion-based VSR practical by achieving **efficiency, scalability, and real-time performance**. To this end, we propose **FlashVSR**, the first diffusion-based one-step streaming framework towards real-time VSR. **FlashVSR runs at ∼17 FPS for 768 × 1408 videos on a single A100 GPU** by combining three complementary innovations: (i) a train-friendly three-stage distillation pipeline that enables streaming super-resolution, (ii) locality-constrained sparse attention that cuts redundant computation while bridging the train–test resolution gap, and (iii) a tiny conditional decoder that accelerates reconstruction without sacrificing quality. To support large-scale training, we also construct **VSR-120K**, a new dataset with 120k videos and 180k images. Extensive experiments show that FlashVSR scales reliably to ultra-high resolutions and achieves **state-of-the-art performance with up to ∼12× speedup** over prior one-step diffusion VSR models.

---

### 🚀 Getting Started

Follow these steps to set up and run **FlashVSR** on your local machine:

> ⚠️ **Note:** This project is primarily designed and optimized for **4× video super-resolution**.  
> We **strongly recommend** using the **4× SR setting** to achieve better results and stability. ✅

#### 1️⃣ Clone the Repository

```bash
git clone https://github.com/cliffpyles/FlashVSR
cd FlashVSR
```

#### 2️⃣ Set Up the Python Environment

**Requirements:**

- Python 3.9 - 3.11 (Python 3.12+ not supported due to torch compatibility)
- [UV](https://docs.astral.sh/uv/) package manager

**Install UV** (if not already installed):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Install project dependencies:**

```bash
# UV will automatically create a virtual environment and install all dependencies
uv sync
```

> **Note:** The project uses `pyproject.toml` for dependency management. UV will automatically:
>
> - Create a virtual environment in `.venv/`
> - Install Python 3.11 (pinned via `.python-version`)
> - Install torch 2.2.2 and all other dependencies
> - Lock dependencies in `uv.lock`

#### 3️⃣ Install Block-Sparse Attention (Required)

FlashVSR relies on the **Block-Sparse Attention** backend to enable flexible and dynamic attention masking for efficient inference.

> **⚠️ Note:**
>
> - The Block-Sparse Attention build process can be memory-intensive, especially when compiling in parallel with multiple `ninja` jobs. It is recommended to keep sufficient memory available during compilation to avoid OOM errors. Once the build is complete, runtime memory usage is stable and not an issue.
> - Based on our testing, the Block-Sparse Attention backend works correctly on **NVIDIA A100 and A800** (Ampere) with **ideal acceleration performance**, and it also runs correctly on **H200** (Hopper) but the acceleration is limited due to hardware scheduling differences and sparse kernel behavior. **Compatibility and performance on other GPUs (e.g., RTX 40/50 series or H800) are currently unknown**. For more details, please refer to the official documentation: https://github.com/mit-han-lab/Block-Sparse-Attention

```bash
# ✅ Recommended: clone and install in a separate clean folder (outside the FlashVSR repo)
git clone https://github.com/mit-han-lab/Block-Sparse-Attention
cd Block-Sparse-Attention
pip install packaging
pip install ninja
python setup.py install
```

#### 4️⃣ Download Model Weights

FlashVSR provides both **v1** and **v1.1** model weights on Hugging Face.  
Model weights are **automatically downloaded** when you first run inference, or you can download them explicitly:

**Option 1: Automatic Download (Recommended)**

```bash
# Models will be auto-downloaded on first use
uv run flashvsr input.mp4
```

**Option 2: Explicit Download**

```bash
# Download v1.1 model (recommended)
uv run flashvsr setup

# Download v1 model
uv run flashvsr setup --version v1

# Download models for specific pipeline
uv run flashvsr setup --pipeline full
uv run flashvsr setup --pipeline tiny
```

Models are downloaded to:

```
./models/FlashVSR/          # v1
./models/FlashVSR-v1.1/     # v1.1
│
├── LQ_proj_in.ckpt
├── TCDecoder.ckpt
├── Wan2.1_VAE.pth
├── diffusion_pytorch_model_streaming_dmd.safetensors
└── README.md
```

> **Note:** The automatic download uses `huggingface_hub` which is faster than Git LFS and supports resume on interrupted downloads. Models are cached locally after the first download. The CLI also checks the old location (`examples/WanVSR/`) for backward compatibility.

---

#### 5️⃣ Run Inference

**Using the CLI (Recommended):**

The CLI is the primary interface for running FlashVSR inference. It automatically handles model downloads and provides a consistent interface.

```bash
# From the repo root
# Basic usage (models auto-download if missing)
uv run flashvsr input.mp4

# Use Tiny pipeline with v1.1 model
uv run flashvsr input.mp4 --pipeline tiny --version v1.1

# Custom output path and scale
uv run flashvsr input.mp4 -o output.mp4 --scale 4.0

# Process image directory
uv run flashvsr ./images/ -o output.mp4

# Customize inference parameters
uv run flashvsr input.mp4 --sparse-ratio 1.5 --local-range 9 --tiled

# See all options
uv run flashvsr --help

# Download models explicitly (optional)
uv run flashvsr setup
```

> **Note:** By default, output videos are saved to the `results/` directory in the project root. Use the `-o` or `--output` option to specify a custom output path.

**Using the example scripts (for reference):**

> **Note:** The example scripts in `examples/WanVSR/` are provided for reference and educational purposes. They demonstrate how to use the FlashVSR pipelines programmatically. For production use, we recommend using the CLI interface above.

```bash
# From the repo root
cd examples/WanVSR

# v1 (original)
uv run python infer_flashvsr_full.py
# or
uv run python infer_flashvsr_tiny.py
# or
uv run python infer_flashvsr_tiny_long_video.py

# v1.1 (recommended)
uv run python infer_flashvsr_v1.1_full.py
# or
uv run python infer_flashvsr_v1.1_tiny.py
# or
uv run python infer_flashvsr_v1.1_tiny_long_video.py
```

> **Note:** Example scripts expect model weights in `examples/WanVSR/FlashVSR/` or `examples/WanVSR/FlashVSR-v1.1/` directories. You can either download models there manually, or create symlinks from the `models/` directory.

---

### 🧪 Testing

FlashVSR includes pytest for testing. To run tests, first install the development dependencies:

```bash
# Install dev dependencies (includes pytest, pytest-cov, pytest-mock)
uv sync --extra dev
```

Then run the test suite:

```bash
# Run all tests
uv run pytest

# Run only unit tests
uv run pytest -m unit

# Run tests with coverage report
uv run pytest --cov=flashvsr

# Run tests in verbose mode
uv run pytest -v
```

Test files are located in the `tests/` directory and follow the naming convention `test_*.py` or `*_test.py`.

---

### 🛠️ Method

The overview of **FlashVSR**. This framework features:

- **Three-Stage Distillation Pipeline** for streaming VSR training.
- **Locality-Constrained Sparse Attention** to cut redundant computation and bridge the train–test resolution gap.
- **Tiny Conditional Decoder** for efficient, high-quality reconstruction.
- **VSR-120K Dataset** consisting of **120k videos** and **180k images**, supports joint training on both images and videos.

<img src="./examples/WanVSR/assets/flowchart.jpg" width="1000" />
