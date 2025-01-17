# ISCO (Intelligent Framework for Segmentation and Comparative Analysis of Organoids)
# ISCO 智能类器官分割与分析算法

ISCO is a powerful tool designed for the automated segmentation and analysis of organoid microscopy images. It combines state-of-the-art deep learning models with comprehensive morphological analysis to provide detailed insights into organoid characteristics.

ISCO 是一个强大的工具，专为类器官显微图像的自动分割和分析而设计。它结合了最先进的深度学习模型和全面的形态学分析，为类器官特征提供详细的见解。

<div class="header">
    <img src="assets/qrcode.png" alt="ISCO QR Code">
    <h1>更多本算法详情欢迎关注肠道类器官微信公众号</h1>
    <p>Intelligent Framework for Accurate Segmentation and Comparative Analysis of Organoids</p>
</div>

## Sponsor | 赞助商

- 上海捷方凯瑞生物科技有限公司 | Shanghai JFKR Biotech Co., Ltd.  
  http://www.jfkrorganoid.cn/
- 吉诺生命健康控股集团有限公司 | Genom Life Health Group Co., Ltd.  
  http://www.genomcell.com/
- ORGARID SCIENTIFIC  | http://www.orgarid.com/

## Features | 功能特点

- 🔬 Automated organoid segmentation using SAM2 (Segment Anything Model 2)  
  使用 SAM2 模型的自动类器官分割
- 📊 Comprehensive morphological analysis including | 全面的形态学分析，包括：
  - Area measurement | 面积测量
  - Perimeter calculation | 周长计算
  - Radius estimation | 半径估计
  - Surface smoothness analysis | 表面平滑度分析
  - Circularity assessment | 圆度评估
  - Symmetry evaluation | 对称性评估
- 📈 Statistical analysis and visualization | 统计分析和可视化
- 🌐 User-friendly GUI interface | 用户友好的图形界面
- 🌍 Multi-language support (English and Chinese) | 多语言支持（中英文）
- 📱 Flexible model size options (tiny, small, base, large) | 灵活的模型大小选项

## Installation Guide | 安装指南

### System Requirements | 系统要求

- Python >= 3.10.0
- CUDA-capable GPU (recommended) | CUDA 兼容的 GPU（推荐）
- Operating System | 支持的操作系统:
  - Windows 10/11
  - Linux (Ubuntu 18.04+)
  - macOS (10.15+)

### Step-by-Step Installation | 分步安装指南

#### 1. Download Project | 下载项目

```bash
git clone https://github.com/yourusername/ISCO.git
cd ISCO
```

#### 2. Choose Installation Method | 选择安装方式

##### Method A: Using Conda (Recommended) | 使用 Conda（推荐）

1. Install Conda | 安装 Conda
   - Download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) or [Anaconda](https://www.anaconda.com/download)
   - 下载并安装 [Miniconda](https://docs.conda.io/en/latest/miniconda.html) 或 [Anaconda](https://www.anaconda.com/download)

2. Create and activate environment | 创建并激活环境
```bash
# Create environment | 创建环境
conda env create -f environment.yml

# Activate environment | 激活环境
conda activate isco
```

##### Method B: Using Pip | 使用 Pip

1. Create virtual environment | 创建虚拟环境
```bash
# Create environment | 创建环境
python -m venv venv

# Activate environment | 激活环境
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate
```

2. Install dependencies | 安装依赖
```bash
# Update pip | 更新pip
python -m pip install --upgrade pip

# Install dependencies | 安装依赖
# Global source | 使用全球源
pip install -r requirements.txt

# Mirror source (China) | 使用国内镜像源
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

#### 3. Download Model Checkpoints | 下载模型文件

1. Download models from | 从以下链接下载模型:  
   链接: https://pan.baidu.com/s/1fGd4x2O5onrUPXWkPK_hYw?pwd=9xjr  
   提取码: 9xjr

2. Place models in checkpoints directory | 将模型文件放入 checkpoints 目录:
   - `checkpoints/sam2.1_hiera_large.pt`
   - `checkpoints/sam2.1_hiera_base_plus.pt`
   - `checkpoints/sam2.1_hiera_small.pt`
   - `checkpoints/sam2.1_hiera_tiny.pt`

### Verification | 验证安装

Run the following command to verify the installation | 运行以下命令验证安装:
```bash
python isco.py --version
```

### Troubleshooting | 常见问题

1. CUDA Issues | CUDA 相关问题
   - Ensure NVIDIA drivers are up to date | 确保 NVIDIA 驱动是最新的
   - Check CUDA compatibility | 检查 CUDA 兼容性

2. Installation Errors | 安装错误
   - Update pip: `python -m pip install --upgrade pip`
   - Try installing packages individually | 尝试单独安装包
   - Check system compatibility | 检查系统兼容性

3. Import Errors | 导入错误
   - Verify environment activation | 确认环境已激活
   - Check package versions | 检查包版本
   - Reinstall problematic packages | 重新安装问题包

For more help | 获取更多帮助:
- Open an issue on GitHub | 在 GitHub 上提交 issue
- Join our WeChat group | 加入我们的微信群

## Usage

### GUI Mode

1. Launch the application:
```bash
python isco.py
```

2. Through the GUI:
   - Select input directory containing PNG images
   - Choose output directory for results
   - (Optional) Provide a mapping file for group analysis
   - Click "Start Processing" to begin analysis

### Command Line Mode

For batch processing, use the command line interface:

```bash
python isco.py -i /path/to/input/directory -o /path/to/output/directory --model large
```

Options:
- `-i, --input`: Input directory containing PNG images
- `-o, --output`: Output directory for results
- `--model`: Model size (tiny/small/base/large)
- `--crop_size`: Size value for cropping image (default: 10000)

## Output

The tool generates:
- Segmentation masks for each image
- CSV files containing morphological measurements
- Statistical analysis reports
- Visualization plots

## Citation

If you use ISCO in your research, please cite:

```
Zhou J, Fu Z, Ni X, et al. ISCO: Intelligent Framework for Accurate Segmentation 
and Comparative Analysis of Organoids[J]. bioRxiv, 2024: 2024.12.24.630244.
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Support

For issues and questions, please open an issue on the Gitee repository. 

## Acknowledgments
We thank the following projects for their contributions to ISCO:
- [Segment Anything Model 2](https://github.com/facebookresearch/segment-anything)
- [Hiera](https://github.com/facebookresearch/Hiera)
- [SAM2](https://github.com/facebookresearch/sam2)
- [SAM](https://github.com/facebookresearch/sam)
