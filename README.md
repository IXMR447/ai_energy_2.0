# 铝液滴燃烧 YOLO-seg 连续帧分析工程

本项目用于分析铝液滴燃烧连续帧图像中的飞溅液滴。核心目标是用训练好的 YOLO-seg 模型识别飞溅液滴实例，提取 mask 几何参数，并在连续帧基础上计算轨迹、速度、断帧补全和轨迹拟合结果。

当前工程重点服务于 30 fps、每组约几十张连续帧的实验图片序列。

## 1. 项目目标

本项目解决的问题是：

```text
输入：一组连续帧燃烧图像
输出：每个飞溅液滴的中心坐标、面积、周长、圆度、等效直径、轨迹 ID、速度、拟合轨迹和可视化结果
```

典型输出包括：

```text
result.csv
result.xlsx
overlay/
tracks.png
```

其中：

```text
overlay/     每张图的 YOLO-seg mask 轮廓、中心点和 track_id 叠加结果
tracks.png   每条液滴轨迹的汇总图
result.csv   适合程序继续分析
result.xlsx  适合人工检查和论文数据整理
```

## 2. 工程目录

推荐目录结构如下：

```text
ai加能源/
  config.yaml
  requirements.txt
  README.md

  weights/
    best.pt

  data/
    组1/
      1.bmp
      2.bmp
      ...
    组2/
    组3/
    组4/

  src/
    run_pipeline.py

  output/
    组1/
      run1.0/
      run2.0/
    组2/
      run1.0/
```

说明：

```text
weights/best.pt   当前使用的 YOLO-seg 模型权重
data/组x          每组待处理图片
output/组x/runx   每次运行的输出版本
config.yaml       默认运行参数
```

## 3. 技术流程

当前技术链路是：

```text
连续帧图片
→ YOLO-seg predict 单帧实例分割
→ 只保留目标类别“飞溅液滴”
→ 从 mask 计算几何参数
→ 基于检测结果做运动匹配轨迹分配
→ 短断帧轨迹重连
→ 缺失帧插值补全
→ 轨迹多项式拟合
→ 速度计算
→ CSV/Excel/overlay/tracks 输出
```

关键设计原则：

```text
检测归检测，轨迹归轨迹。
```

也就是说，`overlay` 和几何参数始终来自 YOLO 原生 `predict()` 结果。轨迹算法只在检测完成后处理 `track_id` 和速度，不再反过来影响单帧识别结果。

之前直接使用 `model.track()` 时，ByteTrack 可能会过滤或改写检测结果，导致 overlay 里部分飞溅液滴不显示。当前版本已经避免这个问题。

## 4. 安装部署

### 4.1 安装 Python 依赖

在项目根目录运行：

```bash
pip install -r requirements.txt
```

主要依赖包括：

```text
ultralytics
opencv-python
numpy
pandas
openpyxl
pyyaml
```

如果没有 NVIDIA GPU，也可以用 CPU 跑，只是速度较慢。对于每组几十张图片的分析，CPU 也可以完成。

### 4.2 放置模型

将训练好的 YOLO-seg 权重放到：

```text
weights/best.pt
```

模型应为 segment 任务，并包含以下类别：

```text
0: 飞溅液滴
1: 主体液滴
```

可以用以下命令快速检查模型：

```bash
python -c "from ultralytics import YOLO; m=YOLO('weights/best.pt'); print(m.task, m.names)"
```

### 4.3 放置数据

例如处理第一组图片：

```text
data/组1/
  1.bmp
  2.bmp
  3.bmp
  ...
```

文件名可以是数字顺序，脚本会按自然顺序排序。

## 5. 快速运行

直接使用 `config.yaml` 中的默认参数运行：

```bash
python src/run_pipeline.py
```

如果要指定某一组输入和输出：

```bash
python src/run_pipeline.py --images "data/组1" --output "output/组1"
```

处理组2：

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2"
```

处理组3：

```bash
python src/run_pipeline.py --images "data/组3" --output "output/组3"
```

每次运行会自动生成版本目录：

```text
output/组1/run1.0/
output/组1/run2.0/
output/组1/run3.0/
```

版本号只在对应组目录内部递增。例如 `output/组2` 会独立从 `run1.0` 开始。

## 6. 配置文件

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
tracker_config: bytetrack.yaml
max_track_distance: 150

max_track_gap: 3
max_gap_link_distance: 220
interpolate_missing: true

fit_tracks: true
track_fit_degree: 2
use_fitted_for_speed: true
min_track_real_points: 3

auto_output_subdir: true
output_version_mode: numeric
output_version_step: 1.0
```

常用参数说明：

```text
conf                    YOLO 置信度阈值。漏检多可降低到 0.10；误检多可提高到 0.20-0.30
imgsz                   推理图像尺寸。小液滴建议 1024 或 1280
target_class            默认只分析飞溅液滴
fps                     帧率，用于速度计算
px_size_mm              像素物理尺寸，例如 1 pixel = 0.01 mm 时填 0.01
tracker                 nearest 表示基于 predict 结果的运动匹配轨迹算法；none 表示不计算轨迹
max_track_distance      相邻匹配的最大距离，单位 pixel
max_track_gap           允许中间漏检的最大帧数
max_gap_link_distance   断帧重连时的最大预测距离，单位 pixel
interpolate_missing     是否补出缺失帧插值点
fit_tracks              是否进行轨迹拟合
track_fit_degree        最大拟合阶数，推荐 2
use_fitted_for_speed    是否使用拟合坐标计算更平滑的速度
min_track_real_points   一条可信轨迹至少需要多少个真实检测点
auto_output_subdir      是否自动生成 run1.0、run2.0 等版本目录
```

## 7. 输出文件说明

每次运行的输出目录类似：

```text
output/组1/run1.0/
  result.csv
  result.xlsx
  summary.csv
  overlay/
  tracks.png
```

### 7.1 result.csv / result.xlsx

主要字段：

```text
frame
frame_index
track_id
class_id
class_name
confidence
xc_px
yc_px
area_px2
perimeter_px
circularity
diameter_px
interpolated
xc_fit_px
yc_fit_px
fit_residual_px
fit_degree
fit_used
vx_px_s
vy_px_s
speed_px_s
vx_fit_px_s
vy_fit_px_s
speed_fit_px_s
```

如果设置了 `px_size_mm`，还会输出：

```text
xc_mm
yc_mm
area_mm2
perimeter_mm
diameter_mm
vx_mm_s
vy_mm_s
speed_mm_s
vx_fit_mm_s
vy_fit_mm_s
speed_fit_mm_s
```

### 7.2 overlay/

每张图片会输出一张叠加图，包含：

```text
YOLO-seg mask
液滴中心点
track_id
圆度 C
```

overlay 的检测结果来自 `model.predict()`，不会被轨迹算法过滤。

### 7.3 summary.csv

每条轨迹一行，用于快速筛查轨迹质量：

```text
track_id
quality
real_points
total_points
frame_start
frame_end
frame_span
mean_confidence
max_fit_residual_px
max_speed_fit_px_s
median_speed_fit_px_s
```

`quality` 包括：

```text
ok              轨迹质量正常
short           真实检测点太少
high_residual   拟合残差过高，可能错连
high_speed      速度峰值过高，需人工检查
```

### 7.4 tracks.png

轨迹汇总图：

```text
彩色线：检测点/插值点形成的轨迹
白色细线：拟合轨迹
```

## 8. 几何参数计算逻辑

YOLO-seg 输出每个液滴的 mask，多边形轮廓用于计算几何参数。

### 8.1 中心坐标

使用轮廓矩计算质心：

```text
xc = M10 / M00
yc = M01 / M00
```

### 8.2 面积

```text
A = mask 轮廓面积
```

单位为 `pixel^2`。

### 8.3 周长

```text
P = mask 轮廓周长
```

单位为 `pixel`。

### 8.4 圆度

```text
C = 4 * pi * A / P^2
```

圆度越接近 1，越接近圆形；越小，说明目标越不规则、拖影越明显或 mask 分割越差。

### 8.5 等效直径

```text
D = sqrt(4 * A / pi)
```

## 9. 轨迹算法逻辑

当前默认轨迹算法为 `nearest`，但它不是简单最近邻，而是基于运动预测的匹配算法。

### 9.1 基本思路

对每一帧的检测结果，按时间顺序处理：

```text
当前帧检测点
→ 与已有轨迹末端进行匹配
→ 根据上一段轨迹预测当前帧位置
→ 计算预测位置和检测点的距离
→ 同时检查目标面积变化是否合理
→ 分配 track_id
```

### 9.2 允许短断帧

如果某个液滴中间 1-3 帧没有被 YOLO 检出，脚本会尝试重连轨迹：

```text
上一段轨迹末端
→ 根据最近运动趋势预测后续位置
→ 与后一段轨迹起点比较
→ 距离小于阈值时合并 track_id
```

相关参数：

```text
max_track_gap
max_gap_link_distance
```

### 9.3 缺失帧插值

当 `interpolate_missing: true` 时，断开的中间帧会补出线性插值点，并用字段标记：

```text
interpolated = true
```

真实 YOLO 检测点为：

```text
interpolated = false
```

正式统计时建议保留这个字段，不要把真实检测点和插值点混在一起解释。

## 10. 轨迹拟合与速度

当 `fit_tracks: true` 时，脚本会对每条轨迹做低阶多项式拟合。

拟合逻辑：

```text
x = f(t)
y = g(t)
```

其中 `t` 是帧序号。默认最大二次拟合：

```text
track_fit_degree: 2
```

如果轨迹点不足，会自动降级：

```text
2 个点：一次线性拟合
3 个及以上点：最多二次拟合
```

拟合输出：

```text
xc_fit_px
yc_fit_px
fit_residual_px
fit_degree
fit_used
```

速度有两套：

```text
vx_px_s, vy_px_s, speed_px_s                  基于原始/插值坐标
vx_fit_px_s, vy_fit_px_s, speed_fit_px_s      基于拟合坐标，更平滑
```

建议做论文或报告时优先检查：

```text
fit_residual_px
interpolated
speed_fit_px_s
```

如果某条轨迹拟合残差过大，说明该轨迹可能存在 ID 错连或检测点异常。

## 11. 常用命令

### 11.1 默认运行

```bash
python src/run_pipeline.py
```

### 11.2 指定组

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2"
```

### 11.3 只看检测，不算轨迹

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2_detect_only" --tracker none --no-auto-output-subdir
```

### 11.4 降低漏检

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2" --conf 0.10 --imgsz 1280
```

### 11.5 误检太多时

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2" --conf 0.20 --imgsz 1280
```

### 11.6 覆盖固定输出目录

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2" --no-auto-output-subdir
```

### 11.7 不筛类别，检查是否分错类

```bash
python src/run_pipeline.py --images "data/组2" --output "output/组2_allclass" --target-class "" --no-auto-output-subdir
```

## 12. 部署建议

### 12.1 本地 Windows 部署

适合当前阶段。

步骤：

```text
1. 安装 Python
2. pip install -r requirements.txt
3. 将模型放到 weights/best.pt
4. 将图片放到 data/组x
5. 修改 config.yaml 或使用命令行参数
6. 运行 python src/run_pipeline.py
```

优点：

```text
部署简单
适合少量图片
方便查看 overlay 和 Excel
```

注意：

```text
如果没有 GPU，推理会慢一些
中文路径在部分第三方库中可能有问题，本项目已对图像读写做兼容处理
```

### 12.2 Linux / GPU 工作站部署

适合后续批量处理和训练。

推荐环境：

```text
Ubuntu 22.04 或 24.04
NVIDIA GPU
Python 3.10+
PyTorch + CUDA
Ultralytics
```

运行方式相同：

```bash
pip install -r requirements.txt
python src/run_pipeline.py --images "data/组1" --output "output/组1"
```

如果使用 GPU：

```bash
python src/run_pipeline.py --device 0
```

### 12.3 云端 API 部署

不推荐作为当前主流程。

原因：

```text
本项目不仅需要 YOLO 预测，还需要 mask 几何计算、轨迹匹配、断帧插值、轨迹拟合和 Excel 输出
云端 API 只适合单张图片预测，不适合完整实验后处理链路
```

更推荐：

```text
训练可以在 Ultralytics 云端完成
推理和分析在本地工程完成
```

## 13. 常见问题

### 13.1 Ultralytics 网页能识别，本地 overlay 看不到

优先检查：

```text
是否筛选了 target_class
conf 是否过高
imgsz 是否过低
是否查看了旧 run 版本
```

可以运行：

```bash
python src/run_pipeline.py --images "data/组2" --output "output/debug_allclass" --target-class "" --tracker none --no-auto-output-subdir
```

### 13.2 小液滴漏检

建议：

```text
conf 降到 0.10
imgsz 提到 1024 或 1280
补充小液滴、暗液滴、模糊液滴训练样本
考虑 ROI 裁剪或灰度 + CLAHE 预处理
```

### 13.3 轨迹断裂

建议：

```text
增大 max_track_gap
增大 max_track_distance
增大 max_gap_link_distance
开启 interpolate_missing
开启 fit_tracks
```

### 13.4 速度跳变

通常原因：

```text
track_id 错连
中间漏检过多
插值点过多
mask 中心点不稳定
```

检查字段：

```text
interpolated
fit_residual_px
speed_px_s
speed_fit_px_s
```

建议优先使用拟合速度：

```text
speed_fit_px_s
```

### 13.5 输出覆盖

默认不会覆盖，会生成：

```text
run1.0
run2.0
run3.0
```

如果确实要覆盖：

```bash
python src/run_pipeline.py --no-auto-output-subdir
```

## 14. 后续可扩展方向

可以继续扩展：

```text
ROI 自动裁剪
灰度 + CLAHE 预处理版本
面积/圆度后处理过滤
异常速度自动剔除
轨迹质量评分
每组实验自动汇总报告
多组结果合并统计
像素标定后输出真实 mm/s 速度
```

当前项目的主线是：

```text
稳定单帧识别
→ 稳定轨迹 ID
→ 合理补全断帧
→ 输出可信的几何参数和速度
```
