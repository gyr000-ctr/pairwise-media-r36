# particles

## 题面（可直接粘贴）

子目录：particles。粒子池与发射曲线。约束：池复用不分配（复用次数可测）；发射速率曲线积分与期望粒子数一致（≤1%）；到期回收且不泄漏；同种子可复现；10 万次更新不超过 300 ms；非法速率可区分报错。验收：cd particles && python -m pytest tests -q。

## 边界要求（评审侧隐藏用例判分）

- 并发安全
- 非有限输入分类报错
- 性能上界按题面

## 运行方式

```bash
cd particles && python -m pytest tests -q
```

初始 15 failed。

## 字段

- 所属领域：游戏、图形与多媒体
- 子方向：游戏开发
- 任务类型：Bugfix
- 语言/框架：Python 3, pytest
- 环境可复现等级：无外部依赖
