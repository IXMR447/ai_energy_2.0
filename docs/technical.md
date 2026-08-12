# 技术文档

本文档说明项目架构、算法流程、参数逻辑、轨迹补全和部署方案。

## 1. 项目定位

本项目是一个面向铝液滴燃烧连续帧图像的智能诊断流程。

核心任务：

```text
YOLO-seg 实例分割
→ mask 几何参数计算
→ 轨迹匹配
→ 断帧补全
→ 轨迹拟合
→ 速度估计
→ 可视化和表格输出
```

设计原则：

```text
检测归检测，轨迹归轨迹。
```

单帧识别结果必须优先保证准确。轨迹算法不能反过来影响 overlay 和几何参数。

## 2. 核心代码

主脚本：

```text
src/run_pipeline.py
```

主要模块：

```text
list_images                  按自然顺序读取图片列表
read_image/write_image        兼容中文路径的图像读写
run_inference                 YOLO-seg predict 推理
detection_from_polygon        从 YOLO mask 构造检测对象
measure_contour               计算面积、周长、圆度、等效直径
assign_tracks_motion          基于运动预测的轨迹匹配
link_tracks_across_gaps       短断帧轨迹重连
add_interpolated_rows         缺失帧插值
add_fitted_track_columns      轨迹多项式拟合
add_track_quality_columns     轨迹质量标记
build_track_summary           轨迹级 summary 输出
draw_overlay                  单帧识别可视化
draw_tracks                   轨迹汇总图
```

## 3. 完整流程

```text
1. 读取 config.yaml 和命令行参数
2. 加载 YOLO-seg 模型 weights/best.pt
3. 按自然顺序读取 data/组x 中的图片
4. 对每张图片执行 model.predict()
5. 根据 target_class 过滤类别
6. 对每个 mask 计算几何参数
7. 基于检测结果分配 track_id
8. 尝试重连短断帧轨迹
9. 对缺失帧做插值补点
10. 对每条轨迹做低阶多项式拟合
11. 计算原始速度和拟合速度
12. 标记轨迹质量
13. 输出 CSV、Excel、summary、overlay 和 tracks
```

## 4. YOLO-seg 推理逻辑

当前版本使用：

```python
model.predict()
```

不使用：

```python
model.track()
```

原因是 `model.track()` 会引入跟踪器状态，可能过滤或改写检测结果。对于本项目，overlay 和 mask 几何参数必须严格来自单帧分割结果。

## 5. 几何参数

每个 YOLO mask 会转换为轮廓。

中心坐标：

```text
xc = M10 / M00
yc = M01 / M00
```

面积：

```text
A = contour area
```

周长：

```text
P = contour perimeter
```

圆度：

```text
C = 4 * pi * A / P^2
```

等效直径：

```text
D = sqrt(4 * A / pi)
```

## 6. 轨迹匹配算法

当前默认：

```yaml
tracker: nearest
```

这里的 `nearest` 是运动预测匹配，不是简单最近邻。

基本逻辑：

```text
对已有轨迹，根据最近两个点估计速度
预测当前帧位置
计算预测位置到当前检测点的距离
检查面积变化是否过大
选择最优匹配
未匹配检测生成新 track_id
```

关键参数：

```text
max_track_distance
max_track_gap
max_gap_link_distance
```

如果错连多：

```text
减小 max_track_distance
减小 max_gap_link_distance
```

如果轨迹太容易断：

```text
增大 max_track_distance
增大 max_track_gap
```

## 7. 断帧重连

YOLO 可能中间漏检 1-3 帧。脚本会尝试将断开的 tracklet 重新连接：

```text
上一段轨迹末端
→ 预测后一段起点位置
→ 距离小于阈值
→ 合并 track_id
```

对应参数：

```yaml
max_track_gap: 3
max_gap_link_distance: 160
```

## 8. 插值补全

当：

```yaml
interpolate_missing: true
```

脚本会在缺失帧生成线性插值点。

新增字段：

```text
interpolated
```

含义：

```text
False   YOLO 实际检测点
True    插值补点
```

正式分析时不要忽略这个字段。

## 9. 轨迹拟合

当：

```yaml
fit_tracks: true
track_fit_degree: 2
```

脚本对每条轨迹拟合：

```text
x = f(t)
y = g(t)
```

点数不足时自动降阶：

```text
2 个点：一次拟合
3 个及以上点：最多二次拟合
```

输出：

```text
xc_fit_px
yc_fit_px
fit_residual_px
fit_degree
fit_used
```

速度输出两套：

```text
speed_px_s       基于原始/插值坐标
speed_fit_px_s   基于拟合坐标，更平滑
```

## 10. 轨迹质量控制

脚本会输出：

```text
track_quality
track_real_points
track_total_points
track_frame_span
```

质量等级：

```text
ok              正常
short           真实检测点太少
high_residual   拟合残差过高
high_speed      速度峰值过高
```

每条轨迹汇总到：

```text
summary.csv
```

建议正式统计时先筛选：

```text
track_quality == ok
```

## 11. 输出版本管理

默认配置：

```yaml
auto_output_subdir: true
output_version_mode: numeric
output_version_step: 1.0
```

输出：

```text
output/组1/run1.0
output/组1/run2.0
```

脚本只扫描当前组目录，因此不同组独立递增。

## 12. 部署方案

### 12.1 Windows 本地

适合当前阶段。

```bash
pip install -r requirements.txt
python src/run_pipeline.py
```

优点：

```text
简单
方便查看 Excel 和 overlay
适合每组几十张图片
```

### 12.2 Linux / GPU 工作站

适合批量推理和后续训练。

推荐：

```text
Ubuntu 22.04/24.04
NVIDIA GPU
Python 3.10+
PyTorch + CUDA
```

运行：

```bash
python src/run_pipeline.py --device 0
```

### 12.3 云端 API

不建议作为主分析流程。

原因：

```text
云端 API 通常只返回 YOLO 预测
本项目还需要几何参数、轨迹、插值、拟合、Excel 输出
完整链路更适合本地或实验室服务器运行
```

推荐方式：

```text
云端训练模型
本地下载 best.pt
本地完成推理和分析
```

## 13. 可优化方向

后续可以继续加入：

```text
ROI 自动裁剪
灰度 + CLAHE 预处理
面积/圆度后处理过滤
主体液滴边缘排除区
异常速度自动剔除
轨迹质量评分更细化
多组实验自动汇总报告
像素标定后输出真实 mm/s
```
