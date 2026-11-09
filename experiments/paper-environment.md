# SEAL / TTRL 的隔离训练环境

论文协议适配使用 [paper-requirements.txt](paper-requirements.txt)，与历史小规模
参考后端的环境分开安装。框架核心仍为零运行依赖；这些包只用于可选 GPU 后端。

已在 B200/B20Z × 8、Python 3.12 下检查 torch 2.7.0+cu128、vLLM 0.9.1、
Transformers 4.52.4、TRL 0.18.1、PEFT 0.15.2、Accelerate 1.7.0、Ray 2.47.1。
先为独立环境安装 CUDA 12.8 构建的 torch/torchvision/torchaudio，再安装依赖：

```bash
uv venv --python 3.12 .local/envs/paper2025
uv pip install --python .local/envs/paper2025/bin/python \
  torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 \
  --index-url https://download.pytorch.org/whl/cu128
uv pip install --python .local/envs/paper2025/bin/python -r experiments/paper-requirements.txt
```

TTRL 还需安装与 Python / torch / CUDA / C++ ABI 匹配的 flash-attn；本机使用
flash-attn 2.8.3、cu12、torch2.7、cxx11abiTRUE 的 cp312 构建。
不能把任意其他 torch 版本的二进制 wheel 混装进来。

本机原始 NCCL 2.26.2 的多卡初始化停滞；以下兼容组合通过了独立 8 卡 NCCL
all-reduce 探针。该组合同时更换库和环境开关，尚未隔离各项的单独作用。
torch 的依赖元数据固定了旧 NCCL，故显式 override 必须在普通依赖安装完成后执行：

```bash
uv pip install --python .local/envs/paper2025/bin/python --no-deps nvidia-nccl-cu12==2.28.9
export NCCL_CUMEM_HOST_ENABLE=0
export NCCL_CUMEM_ENABLE=0
export NCCL_NVLS_ENABLE=0
```

后续重新解析环境可能恢复 torch 固定的旧 NCCL；先记录实际环境版本，再运行训练。
框架会把原生 TTRL 的 NCCL/VLLM 开关保存到运行目录，不修改机器的系统环境。
每次 TTRL fit 启动自己持有的 Ray 集群，退出时关闭该集群。

在两份 `paper.json` 中分别配置模型、`training_python` 和 `inference_python` 的
实际路径。TTRL 的 `upstream_root` 必须是文档指定 revision 的干净 checkout。
SEAL 的语义判分额外需要 `OPENAI_API_KEY`；凭据通过进程环境传入，不能写入配置或日志。
运行步骤见 [SEAL](seal/README.md) 和 [TTRL](ttrl/README.md)。
