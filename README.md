# 铝液滴燃烧 YOLO-seg 连续帧分析工程

本项目用于分析铝液滴燃烧连续帧图像中的飞溅液滴，输出液滴 mask、中心坐标、面积、圆度、轨迹、速度、断帧补全和轨迹拟合结果。

文档已拆分为两部分：

```text
docs/usage.md      使用文档：安装、放模型、放数据、运行命令、查看输出、常见操作
docs/technical.md  技术文档：项目架构、算法流程、参数逻辑、轨迹拟合、部署方案
```

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

详细使用说明见 [docs/usage.md](docs/usage.md)。

技术架构说明见 [docs/technical.md](docs/technical.md)。
