# outline

## 题面（可直接粘贴）

子目录：outline。描边网格生成。约束：沿法线外扩（不改变原网格）；斜接/圆角两种连接可配；内角处自交必须修正（面积与外扩量一致 ≤1e-6）；闭合轮廓首尾缝合；1 万顶点不超过 700 ms；非法宽度可区分报错。验收：cd outline && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd outline && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：图形渲染
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
