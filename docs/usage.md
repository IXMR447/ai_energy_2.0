# 使用文档

本文档面向日常使用：如何安装环境、放置模型、导入数据、运行程序、查看输出和调整常用参数。

## 1. 安装依赖

在项目根目录运行：

```bash
pip install -r requirements.txt
```

主要依赖：

```text
ultralytics
opencv-python
numpy
pandas
openpyxl
pyyaml
```

如果没有 NVIDIA GPU，也可以用 CPU 跑，只是速度较慢。当前每组几十张图片，CPU 也可以处理。

## 2. 放置模型

将训练好的 YOLO-seg 权重放到：

```text
weights/best.pt
```

检查模型：

```bash
python -c "from ultralytics import YOLO; m=YOLO('weights/best.pt'); print(m.task, m.names)"
```

正常应类似：

```text
segment {0: '飞溅液滴', 1: '主体液滴'}
```

## 3. 放置数据

每组图片放在一个文件夹：

```text
data/组1/
  1.bmp
  2.bmp
  ...

data/组2/
data/组3/
data/组4/
```

脚本会按自然顺序读取图片。

## 4. 运行程序

使用 `config.yaml` 默认配置：

```bash
python src/run_pipeline.py
```

指定组1：

```bash
python src/run_pipeline.py --images "data/组1" --output "output/组1"
```

指定组2：

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2"
```

指定组3：

```bash
python src/run_pipeline.py --images "data/组3" --output "output/组3"
```

## 5. 输出目录

默认不会覆盖旧结果。每次运行会自动生成版本文件夹：

```text
output/组1/run1.0/
output/组1/run2.0/
output/组1/run3.0/
```

不同组独立递增：

```text
output/组2/run1.0/
output/组2/run2.0/
```

如果要覆盖固定目录：

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2" --no-auto-output-subdir
```

## 6. 输出文件

每个 run 目录包含：

```text
result.csv
result.xlsx
summary.csv
overlay/
tracks.png
```

说明：

```text
result.csv    每帧每个液滴一行，适合程序分析
result.xlsx   Excel 版本，适合人工检查
summary.csv   每条轨迹一行，适合筛查轨迹质量
overlay/      每张图的分割轮廓、中心点和 ID 叠加
tracks.png    所有轨迹汇总图
```

## 7. 常用参数

默认参数在 `config.yaml` 中：

```yaml
images: data/组1
output: output/组1
model: weights/best.pt

fps: 30
px_size_mm:

target_class: 飞溅液滴
conf: 0.10
imgsz: 1280
device:

tracker: nearest
max_track_distance: 100
max_track_gap: 3
max_gap_link_distance: 160
interpolate_missing: true

fit_tracks: true
track_fit_degree: 2
use_fitted_for_speed: true
min_track_real_points: 3

auto_output_subdir: true
output_version_mode: numeric
output_version_step: 1.0
```

最常调：

```text
conf                  漏检多就降低，误检多就提高
imgsz                 小液滴建议 1024 或 1280
max_track_distance    轨迹容易断可增大，错连多就减小
max_track_gap         允许中间漏检几帧
px_size_mm            像素标定后可输出 mm 和 mm/s
```

## 8. 常用命令

降低漏检：

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2" --conf 0.10 --imgsz 1280
```

误检太多：

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2" --conf 0.20
```

只看 YOLO 检测，不算轨迹：

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2_detect_only" --tracker none --no-auto-output-subdir
```

不筛类别，用来检查是否分错类：

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2_allclass" --target-class "" --no-auto-output-subdir
```

使用 GPU：

```bash
python src/run_pipeline.py --device 0
```

## 9. 查看结果

建议顺序：

```text
1. 看 overlay/，确认每帧液滴识别是否准确
2. 看 tracks.png，确认轨迹是否有明显错连
3. 看 summary.csv，筛查 short/high_residual/high_speed 轨迹
4. 看 result.xlsx，检查具体坐标、速度、插值点
```

正式统计时建议优先使用：

```text
track_quality == ok
speed_fit_px_s 或 speed_fit_mm_s
interpolated 字段用于区分真实点和补点
```

## 10. 常见问题

### 10.1 网页能识别，本地看不到

检查：

```text
是否筛选了 target_class
conf 是否过高
imgsz 是否过低
是否查看了旧 run 目录
```

### 10.2 小液滴漏检

建议：

```text
conf 降到 0.10
imgsz 提到 1024 或 1280
补充小液滴、暗液滴、模糊液滴训练样本
```

### 10.3 轨迹错连

建议：

```text
减小 max_track_distance
减小 max_gap_link_distance
检查 summary.csv 中 high_residual 和 high_speed
```

### 10.4 速度跳变

优先检查：

```text
track_quality
fit_residual_px
interpolated
speed_fit_px_s
```

建议统计时先过滤：

```text
track_quality == ok
```
