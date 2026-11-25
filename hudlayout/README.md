# hudlayout

## 题面（可直接粘贴）

子目录：hudlayout。HUD 布局约束求解。约束：按锚点/边距/尺寸约束求位置（与线性约束解一致 ≤1e-9）；重叠最小化且优先级可配；屏幕缩放后布局按安全区重算；循环依赖可区分报错；1 万控件不超过 700 ms；非法锚点可区分报错。验收：cd hudlayout && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd hudlayout && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
