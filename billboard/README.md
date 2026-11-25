# billboard

## 题面（可直接粘贴）

子目录：billboard。公告板朝向与排序。约束：朝向按相机基（屏幕对齐/轴向/球形三种模式）计算且保持正交（≤1e-9）；排序按视深稳定（同深度保持插入序）；背面剔除判定与点积一致；1 万公告板不超过 400 ms；非法模式可区分报错。验收：cd billboard && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd billboard && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
