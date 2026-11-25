"""billboard 核心 API（初始未实现）。"""


class Billboard:
    def orient(self, pos, cam_fwd, cam_up, mode):
        """朝向"""
        raise NotImplementedError

    def sort(self, items, cam_pos):
        """排序"""
        raise NotImplementedError
