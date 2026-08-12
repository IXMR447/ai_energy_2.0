<h1 align="center">铝液滴燃烧 YOLO-seg 连续帧分析工程</h1>

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/YOLO--seg-Ultralytics-111F68?style=flat-square" alt="YOLO-seg">
  <img src="https://img.shields.io/badge/Task-Instance%20Segmentation-00A67E?style=flat-square" alt="Instance Segmentation">
  <img src="https://img.shields.io/badge/Tracking-Motion%20Matching-FFB000?style=flat-square" alt="Tracking">
  <img src="https://img.shields.io/badge/Output-CSV%20%7C%20Excel%20%7C%20Overlay-4B5563?style=flat-square" alt="Output">
</p>

<p align="center">
  <img src="https://img.shields.io/badge/Domain-Aluminum%20Droplet%20Combustion-D97706?style=for-the-badge" alt="Domain">
  <img src="https://img.shields.io/badge/Workflow-Predict%20%E2%86%92%20Measure%20%E2%86%92%20Track%20%E2%86%92%20Fit-2563EB?style=for-the-badge" alt="Workflow">
</p>

本项目用于分析铝液滴燃烧连续帧图像中的飞溅液滴，输出液滴 mask、中心坐标、面积、圆度、轨迹、速度、断帧补全和轨迹拟合结果。

## 文档入口

| 文档 | 内容 |
| --- | --- |
| [使用文档](docs/usage.md) | 安装、放模型、放数据、运行命令、查看输出、常见操作 |
| [技术文档](docs/technical.md) | 项目架构、算法流程、参数逻辑、轨迹拟合、部署方案 |
| [预处理建议](docs/image_preprocessing_recommendations.md) | ROI 裁剪、灰度、CLAHE、去噪、光晕干扰处理 |

## 功能标签

<p>
  <img src="https://img.shields.io/badge/%E5%8D%95%E5%B8%A7%E8%AF%86%E5%88%AB-YOLO%20predict-22C55E?style=flat-square" alt="单帧识别">
  <img src="https://img.shields.io/badge/mask%20%E6%B5%8B%E9%87%8F-%E9%9D%A2%E7%A7%AF%20%7C%20%E5%91%A8%E9%95%BF%20%7C%20%E5%9C%86%E5%BA%A6-0EA5E9?style=flat-square" alt="mask测量">
  <img src="https://img.shields.io/badge/%E8%BD%A8%E8%BF%B9-ID%20%E5%8C%B9%E9%85%8D-A855F7?style=flat-square" alt="轨迹">
  <img src="https://img.shields.io/badge/%E6%96%AD%E5%B8%A7-%E6%8F%92%E5%80%BC%E8%A1%A5%E5%85%A8-F59E0B?style=flat-square" alt="断帧补全">
  <img src="https://img.shields.io/badge/%E9%80%9F%E5%BA%A6-%E6%8B%9F%E5%90%88%E5%B9%B3%E6%BB%91-EF4444?style=flat-square" alt="速度">
  <img src="https://img.shields.io/badge/%E8%B4%A8%E9%87%8F-summary.csv-64748B?style=flat-square" alt="质量控制">
</p>

## 快速运行

安装依赖：

```bash
pip install -r requirements.txt
```

放置模型：

```text
weights/best.pt
```

放置数据：

```text
data/组1/
data/组2/
data/组3/
```

运行默认配置：

```bash
python src/run_pipeline.py
```

指定某一组：

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2"
```

输出示例：

```text
output/组2/run1.0/
  result.csv
  result.xlsx
  summary.csv
  overlay/
  tracks.png
```

## 技术流程

```mermaid
flowchart LR
    A["连续帧图片"] --> B["YOLO-seg predict"]
    B --> C["mask 几何测量"]
    C --> D["运动匹配 track_id"]
    D --> E["断帧重连与插值"]
    E --> F["轨迹拟合与速度计算"]
    F --> G["CSV / Excel / overlay / tracks"]
```

## 输出概览

| 文件 | 作用 |
| --- | --- |
| `result.csv` | 每帧每个液滴的完整参数，适合后续程序分析 |
| `result.xlsx` | Excel 结果，适合人工检查 |
| `summary.csv` | 每条轨迹的质量汇总，适合筛选异常轨迹 |
| `overlay/` | 每张图的分割轮廓、中心点、ID 叠加结果 |
| `tracks.png` | 所有轨迹汇总图，彩色为轨迹，白色为拟合线 |

## 推荐阅读顺序

1. 第一次运行：先看 [使用文档](docs/usage.md)
2. 调参数和部署：看 [技术文档](docs/technical.md)
3. 处理漏检、光晕、弱液滴：看 [预处理建议](docs/image_preprocessing_recommendations.md)
