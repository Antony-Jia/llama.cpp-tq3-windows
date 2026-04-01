# llama.cpp-tq3

[English Version (英文版本)](README.md)

**在单张 16GB GPU 上运行 27B 大语言模型 — 质量接近 Q4_0。**

TQ3_1S 是一种 3.5-bit 量化格式，使用 Walsh-Hadamard 变换和双尺度编码压缩模型权重。在 Qwen3.5-27B 上，它达到了接近 `Q4_0` 的质量，同时模型体积更小 —— 使得 `Q4_0` 无法完全放入的消费级显卡也能够完整容纳。

这个项目是 [llama.cpp](https://github.com/ggml-org/llama.cpp) 的分支，添加了运行 TQ3_1S 模型所需的全速 CUDA 内核。

## 为什么这很重要

大语言模型强大但运行成本高。27B 参数模型使用 `Q4_0` 量化后体积仍然较大，在 16GB 显卡上无法完整放入显存。

TQ3_1S 解决了这个部署问题。它将相同模型压缩到约 12.9 GB，同时保持接近 `Q4_0` 的质量，能够完整放入 16GB 显存。与部分卸载到 CPU 的 `Q4_0` 相比，端到端吞吐量明显提升。

## 性能测试 (Qwen3.5-27B, RTX 5060 Ti 16GB)

使用标准 `wiki.test.raw` 测试，`c=512`，完整 `580` 块：

| | TQ3_1S | Q4_0 |
|---|---:|---:|
| 困惑度 (Perplexity) | `7.2570 +/- 0.0480` | `7.2431 +/- 0.0482` |
| 模型体积 | **12.9 GB** | 14.4 GB |
| 可完整放入 16GB GPU | **✅** | ❌ |

困惑度差距: `+0.0139`，约 **0.19%**。

合理的解读：

- `TQ3_1S` 相比 `Q4_0` 体积明显更小
- `TQ3_1S` 在 27B 模型上达到了接近 `Q4_0` 的质量
- 实际速度提升来自**在 16GB GPU 上的完整部署优势**，这并不意味着原生 `TQ3_1S` 内核普遍比原生 `Q4_0` 更快

## 预量化模型

**[Hugging Face 上的 turbo-tan/Qwen3.5-27B-TQ3_1S](https://huggingface.co/turbo-tan/Qwen3.5-27B-TQ3_1S)** —— 下载即可运行。

## 从源码编译

```bash
git clone https://github.com/turbo-tan/llama.cpp-tq3.git
cd llama.cpp-tq3

cmake -B build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Windows 编译问题修复

在 Windows 上使用 CUDA 编译时，可能会遇到以下两个问题，这里记录解决方法：

### 问题 1: `ggml_cuda_op_turbo_wht was referenced but not defined`

**原因**: 如果你先运行 `cmake` 再运行 `git submodule update --init --recursive`，CMake 在第一次配置时 `turbo-wht.cu` 文件还不存在，因此不会将其加入编译列表。

**解决方法**: 在 `git submodule update` 完成后，重新运行 CMake 并清理缓存：
```powershell
cd llama.cpp-tq3
# 重新配置
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
# 清理旧编译缓存
rm -rf build/ggml/src/ggml-cuda/ggml-cuda.dir/Release/*
# 重新编译
cmake --build build --config Release --target llama-cli llama-server -j 8
```

### 问题 2: `identifier "ggml_tq3_ap_extra" is undefined`, `GGML_TQ3_AP_MAGIC: undeclared identifier`

**原因**: `llama-model.cpp` 中使用了 `ggml_tq3_ap_extra` 结构体但缺少头文件声明。

**解决方法**: 在 `src/llama-model.cpp` 开头添加以下声明：
```cpp
// TurboQuant TQ3 activation quantization extra data
struct ggml_tq3_ap_extra {
    int magic;
    float * means;
    int * row_offsets;
    uint8_t * bitmap;
};

#define GGML_TQ3_AP_MAGIC 0x54513341 // "TQ3A"
```

### 完整的 Windows 编译命令（RTX 40xx 系列）

```powershell
cd D:\LlmModels\Qwen3.5-27B-TQ3_1S
git clone --branch main https://github.com/turbo-tan/llama.cpp-tq3.git llama.cpp-tq3-main
cd .\llama.cpp-tq3-main
git submodule update --init --recursive
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --config Release --target llama-cli llama-server -j 8
```

修复以上两个问题后，即可成功编译出 `llama-cli.exe` 和 `llama-server.exe`。

## 快速开始

设置路径变量：

```bash
export USERNAME="${USERNAME:-$USER}"
export CODE_ROOT="/home/$USERNAME/code"
export MODEL_ROOT="/home/$USERNAME/models"
```

从 Hugging Face 下载 GGUF：

```bash
mkdir -p "$MODEL_ROOT/turboquant27"

python3 - <<'PY'
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="turbo-tan/Qwen3.5-27B-TQ3_1S",
    filename="Qwen3.5-27B-TQ3_1S.gguf",
    local_dir=f"/home/{__import__('os').environ.get('USERNAME', __import__('os').environ.get('USER'))}/models/turboquant27",
)
PY
```

```bash
# 启动已发布的 27B 模型服务
./build/bin/llama-server \
    -m "$MODEL_ROOT/turboquant27/Qwen_Qwen3.5-27B-TQ3_1S.gguf" \
    -ngl 99 \
    -fa on \
    -c 4096 \
    --port 8090 \
    --reasoning off \
    --reasoning-budget 0 \
    --reasoning-format none
```

## 性能测试

标准困惑度测试：

```bash
./build/bin/llama-perplexity \
    -m "$MODEL_ROOT/turboquant27/Qwen_Qwen3.5-27B-TQ3_1S.gguf" \
    -f "$CODE_ROOT/llama.cpp/wikitext-2-raw/wiki.test.raw" \
    -c 512 \
    -ngl 99 \
    -fa 1 \
    -t 8 \
    --no-warmup
```

快速吞吐量测试：

```bash
./build/bin/llama-bench \
    -m "$MODEL_ROOT/turboquant27/Qwen_Qwen3.5-27B-TQ3_1S.gguf" \
    -ngl 99 \
    -fa on
```

如果你从 Hugging Face 下载到了其他位置，相应修改 `MODEL_ROOT` 即可。

## TQ3_1S 工作原理

标准量化将每个权重量化到均匀网格上的最近值。TQ3_1S 采用不同的方法：

1. **旋转** — 对每个 32 元素权重块应用 Walsh-Hadamard 变换。这将信息分散到各个元素，使分布更加均匀，更容易量化。灵感来自 [RaBitQ](https://arxiv.org/abs/2405.12497)。

2. **量化** — 将每个旋转后的值映射到 8 个学习到的中心点之一（3 bits）。这些中心点针对旋转后的分布进行了优化。

3. **双尺度** — 每个块存储两个 fp16 缩放因子：一个用于 0–15 元素，一个用于 16–31 元素。这种方式能够捕捉单个缩放因子会丢失的局部幅度变化。

块布局：`[d0: fp16][d1: fp16][qs: 12 bytes]` = 每个 32 元素占 16 字节（每个权重 4.0 bits）

推理过程中，激活会预先旋转到相同的 WHT 域，允许 CUDA 内核直接对中心点计算点积，而不需要对权重进行逆变换。

## 项目历程

这个项目源于一个实际需求：在消费级硬件上运行可用的模型，不需要租用云 GPU。我们探索了自适应块提升、imatrix 加权量化、混合精度混合和多种变换想法。其中许多方法改进了张量指标，但在模型级别无法保持效果。双尺度 WHT 方法留存下来，因为它在保持运行时路径足够简单可部署的同时，提供了最强的实际 27B 结果。

为 token 生成提供支持的 MMVQ 内核经过了 CPU 基线验证，然后在完整的 `580` 块 `wiki.test.raw` 测试上进行压力测试，以捕捉短评估可能遗漏的数值漂移。

## 验证

头条声明所使用的记录制品：

- 完整 `TQ3_1S` PPL 测试: `7.2570 +/- 0.04802`
- 完整 `Q4_0` PPL 测试: `7.2431 +/- 0.04822`
- 模型体积:
  - `TQ3_1S`: 约 `12.9 GB`
  - `Q4_0`: 约 `14.4 GB`

如果你想复现结果，请使用相同的基础模型、相同的 `wiki.test.raw` 语料库和相同的 `llama-perplexity` 设置在两种格式上测试。

## 致谢

- [RaBitQ](https://arxiv.org/abs/2405.12497) — Walsh-Hadamard 变换灵感
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — 推理引擎
- [Qwen3.5-27B](https://huggingface.co/Qwen/Qwen3.5-27B) — 基础模型

## 许可证

MIT — 与 llama.cpp 相同
