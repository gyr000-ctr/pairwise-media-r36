"""outline 核心 API（初始未实现）。"""


class Outline:
    def build(self, poly, width, join='miter'):
        """生成描边"""
        raise NotImplementedError

    def area(self, poly):
        """面积"""
        raise NotImplementedError
