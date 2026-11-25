"""particles 核心 API（初始未实现）。"""


class Particles:
    def __init__(self, cap, seed):
        """构造"""
        raise NotImplementedError

    def emit(self, rate, dt):
        """发射"""
        raise NotImplementedError

    def update(self, dt):
        """更新"""
        raise NotImplementedError

    def live(self):
        """存活数"""
        raise NotImplementedError
