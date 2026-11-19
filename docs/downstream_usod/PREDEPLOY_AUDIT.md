# ICAR-UIE × TC-USOD 部署前资产核查

主项目 Git 根为 UIE4_VN，origin 为 Panke-Git/UIE4_VN，基线提交是 b5c9a77da721b03ee777ff75dbbe2872d02baf03。Mac 主工作树已有色彩分析脚本改动和未跟踪 outputs。本功能在独立 worktree 上的 feat/downstream-usod-tc-usod 开发。FX 在工作区中是主项目的同级目录，本次 Git 发布范围是 UIE4_VN。

本次 SSH 只读核查发现 school 有 RTX 3090 24GB，原 UIE4_VN 工作树已有本地脚本改动，预期位置缺 publicdata、FX 和两套正式 V16 run。服务器部署使用新的干净检出目录，并将实验结果写入约定目录。home-wsl 的现有转发本次拒绝连接，以下正式 V16 的历史位置来自此前存档。

## 作者代码、数据及权重

| 资产 | 来源与核查 |
| --- | --- |
| TC-USOD 源码 | [作者 Git 仓库](https://github.com/LinHong-HIT/USOD10K/tree/fcdd53926054c79c20a64d1c69621316c5562929) 固定 fcdd53926054c79c20a64d1c69621316c5562929。运行时校验被导入的作者 Models Python 文件。 |
| 许可证 | 固定的作者仓库树中未见项目级 LICENSE。T2T-ViT 部分源码有 Clear BSD 声明。通过独立作者 checkout 使用其模型代码。README 将数据限定为学术研究使用。 |
| TC-USOD 全模型 | [作者 Google Drive baseline](https://drive.google.com/file/d/1fFKhuuR2MEEjBWRtjFdMZCoRxLrQqUK1/view)，官方 README 也有百度网盘链接。本次实际下载得到 TCUSOD.pth，504,324,899 字节。 |
| 全模型 SHA-256 | a17524c8374b60c12c7c81e87b3ebfe657ec49c4f8b5cfeedbb82eb77e2b0c29。本地严格加载 125,957,743 个参数，源 state_dict 采用 module. 前缀。 |
| 数据 | [作者 USOD10K Google Drive](https://drive.google.com/file/d/1PH0PwKchXnkWwtAwbhNSW4utMCp5zer8/view)。公开 USOD10k.zip 大小 5,665,725,063 字节，实际打包结构为 USOD10k/USOD10k/TE/{RGB,depth,GT}.zip。 |
| 官方 Test 成员 | 从作者发布的 TC_USOD/preds/USOD10K 的 1,026 个文件名提取 ID，再与三个官方 Test 子包逐项匹配。仓库根 test.txt 内容为占位文本。排序 ID 每行一条、末尾含换行，其 SHA-256 为 e1cb352150706d388c4683abf9b0ae0279fe36d75dd9b964280abb5c4cd8aa92。 |
| 实际数据 | 每模态 1,026 张 PNG、640×480。RGB 为 RGB8，Depth 为单通道 16 位。GT 原文件为 1,025 张 RGB 和 1 张 RGBA，转成灰度后 273 张含 0/255 以外像素。数据准备保留官方原文件字节。 |
| Mac 可迁移目录 | 工作区 publicdata/USOD10K/USOD10K-TE 下三个模态文件夹，以及 publicdata/USOD10K_assets/TCUSOD.pth。独立作者 checkout 保存在 publicdata/USOD10K_assets/USOD10K-source，AutoDL 可按部署文档复取。 |

已提取的 Test 子包 SHA-256 如下。

| 子包 | SHA-256 |
| --- | --- |
| RGB.zip | 3b2fba3b575f1a1a591db75df09631fd77b925140b300ab2b0f5a1cde6f568ea |
| depth.zip | 2965ddee8f87cb76bc9413561c2232877941bf0f40e1d377f1d7a9f5a3b0ebf9 |
| GT.zip | dab0755d1dcbe215bac28c6cece402549d93eb500fd9da0f31937e4fe4cd4298 |

在作者 README 命名的三个子目录下重排实际解包文件后，本地完整 manifest SHA-256 为 8da85da3fdba39b0c18a612c56b010c7c3f4bd599324360a6e0bbf9ed2ccd15b。该摘要包含相对路径、样本尺寸及每个文件的 SHA-256。改用其他相对目录名会产生对应的 manifest 摘要。

作者推理使用官方 Testing.py、dataset.py、transforms.py 及 Evaluation/evaluator.py。本项目用严格全模型加载覆盖全部初始化参数，构造时跳过重复的 T2T-ViT 与 ResNet ImageNet 初始化。作者仓库的 80.7_T2T_ViT_t_14.pth.tar 只用于 backbone 初始化。RGB 与 16 位 Depth 均沿用作者的 Pillow RGB 转换，再进入相同的双线性缩放和 ImageNet 归一化。保留 d1 的量化和原尺寸插值顺序。 [固定的作者源文件](https://github.com/LinHong-HIT/USOD10K/tree/fcdd53926054c79c20a64d1c69621316c5562929/TC_USOD)

## 项目权重和 FX 迁移项

两个正式 V16 run 的历史父目录是 WSL 上的 /home/paxton/pro/UIE4_VN/experiments。每个启用的 run 迁移 config_resolved.yaml 与 best/best_psnr.pt。V16 使用现有 src.v16.models.network.build_model 并严格比较模型配置、epoch 和权重。USOD 冻结推理只需要其下游输入。

| 阶段 | 历史 run | 验证选择与 checkpoint SHA-256 |
| --- | --- | --- |
| P0、P1 | v16_PlainUNet_ColorQuery_UICF_PreBackbone_seed3520_20260824_004528 | LSUI Val PSNR，epoch 195，ae0136c6fbb539d44caec25e7e35579cda50c3cadc0e6532de861c82a19e0ab4 |
| P2 | v16_PlainUNet_ColorQuery_UICF_PreBackbone_UIEB_T90_seed3520_20260825_232959 | UIEB Val PSNR，epoch 49，781ea7923ed328ad1491d353abbef1aeed932b52e6f0ff2927fdbcf4d175aa12 |

Mac 未找到与两个正式 SHA-256 对应的本地 run 文件。当前 AutoDL 也缺这些 run。Mac 上已有 DUO 的合成 V16 fixture。此次本地增强集成使用专门的合成模型结构测试，正式加载器始终要求上表权重。

P1 通过独立 Python 进程调用每个 FX 方法的原生推理函数，重用其 U45 输入适配中的模型、前处理和 PNG8 输出逻辑。USOD 使用自己的 manifest 和显著性评价器。每个启用方法迁移其目录、FX/_shared、正式 checkpoint 和兼容解释器，并在 YAML 中填写来源和 SHA-256。已有 LSUI/UIEB/U45 的增强预测不能替代 USOD 输入上的增强推理。

| FX 方法 | 复用接口 | 需迁移的 Val 选中权重 |
| --- | --- | --- |
| WaterNet | adapters.infer_u45 的原生 RGB/WB/HE/Gamma 输入及模型 | best_psnr.pt |
| Ucolor | adapters.infer_u45 的 GDCP、原生模型 padding/crop 及生产宽度 | best_psnr.pt |
| FDCE-Net | adapters.infer_u45 的生产模型与 forward_padded | best_ssim.pt |
| MARINE-Net | adapters.inference 的原生 predict 和正式 pipeline checkpoint 校验 | best_model.pth |
| SWLGM | adapters.u45 的模型与 native_size 路径，解释器需要可用的 Mamba CUDA 栈 | best_psnr.pt |

本地审计确认五个 FX 代码入口存在，正式 FX 资产尚待迁移并通过各自的真实 smoke。已有历史重评分可用于确认方法和训练域来源。
