# AMD-MI300X-Guide

## Sglang x DeepSeek V3/R1

### Step 1: Create Conda Environment

```bash
conda create -n sglang python=3.10 -y
```

### Step 2: Install Torch

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm6.2.4
```

### Step 3: Install vLLM

```bash
git clone https://github.com/vllm-project/vllm.git
cd vllm
git checkout v0.6.6
sudo ~/miniconda3/envs/sglang/bin/pip install /opt/rocm/share/amd_smi
pip install --upgrade numba scipy "huggingface-hub[cli]"
pip install "numpy<2"
pip install setuptools_scm
pip install "cmake>=3.26.0"
pip install -r requirements-rocm.txt
export PYTORCH_OCM_ARCH="gfx942"
export CCACHE_DIR=$HOME/.cache/ccache
python3 setup.py develop
```

> [!NOTE]
> It is recommended to use vLLM version v0.6.6 and avoid versions >= v0.7.0.
>
> `sudo pip` is actually `/usr/bin/pip`, please use `~/miniconda3/envs/sglang/bin/pip` to install `amd_smi`.

### Step 4: Install Triton

```bash
git clone https://github.com/triton-lang/triton.git
cd triton
git checkout 845d75a
cd python
python3 setup.py install
```

### Step 5: Install ater

```bash
git clone https://github.com/HaiShaw/ater.git
cd ater
git submodule update --init --recursive
PREBUILD_KERNELS=1 GPU_ARCHS=gfx942 python3 setup.py develop
```

### Step 6: Install SGLang

```bash
# Use the last release branch
git clone -b v0.4.2.post3 https://github.com/sgl-project/sglang.git
cd sglang
cd sgl-kernel
python setup_rocm.py install
cd ..

# Update pyproject.toml in python/ before next step
# Replace:
# srt_hip = ["sglang[runtime_common]", "torch", "vllm==0.6.7.dev2", "outlines==0.1.11", "sgl-kernel>=0.0.3.post1"]
# With:
# srt_hip = ["sglang[runtime_common]", "outlines==0.1.11", "sgl-kernel>=0.0.3.post1"]

pip install -e "python[all_hip]"

# If you are in the docker (you may encounter the config file name issue)
cd python/sglang/srt/layers/quantization/configs/
for file in *.json; do
    N=$(echo "$file" | grep -oP 'N=\K\d+')
    K=$(echo "$file" | grep -oP 'K=\K\d+')
    dtype=$(echo "$file" | grep -oP 'dtype=\K[^,]+')
    block_shape=$(echo "$file" | grep -oP 'block_shape=\K[^.]+')
    new_file="N=${N},K=${K},device_name=AMD_Instinct_MI300X_VF,dtype=${dtype},block_shape=${block_shape}.json"
    cp "$file" "$new_file"
done
```

> [!TIP]
> If you encounter the GLIBCXX error, run `conda install -c conda-forge gcc gxx libstdcxx-ng`.

### Step 7: Dive with DeepSeek V3/R1

```bash
pip install hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1
huggingface-cli download deepseek-ai/DeepSeek-R1 --local-dir ~/models/DeepSeek-R1
python -m sglang.launch_server --model-path ~/models/DeepSeek-R1 --tp 8 --trust-remote-code
```

> [!IMPORTANT]
> After installing the `hf_transfer` package, please reactivate the conda environment then export the `HF_HUB_ENABLE_HF_TRANSFER` variable.