class Encoder(object):
    def __call__(self, data):
        raise NotImplementedError


class IdentityEncoder(Encoder): # 恒等映射编码器，直接返回输入数据不做任何变换
    def __call__(self, data):
        return data


class SMAC5m6mEncoder(Encoder): # SMAC5m6m环境的编码器，将历史状态复制到当前时间步
    def __call__(self, data):
        data[..., 1:, :, :5] = data[..., 0:1, :, :5]
        return data


class SMAC3mEncoder(Encoder): # SMAC3m环境的编码器，将历史状态复制到当前时间步
    def __call__(self, data):
        data[..., 1:, :, :3] = data[..., 0:1, :, :3]
        return data
