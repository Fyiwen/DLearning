from torch import nn

# 设计通用的编码器-解码器结构
# 在编码器接口中，只指定长度可变的序列作为编码器的输入X。 任何继承这个Encoder基类的模型将完成代码实现
class Encoder(nn.Module):
    """编码器-解码器架构的基本编码器接口"""
    def __init__(self, **kwargs): # Encoder到时候由哪个网络模型实现，都行
        super(Encoder, self).__init__(**kwargs) # 继承到时候具体的网络模型参数初始化

    def forward(self, X, *args):
        raise NotImplementedError # 表示该部分的功能尚未被实现

class Decoder(nn.Module):
    """编码器-解码器架构的基本解码器接口"""
    def __init__(self, **kwargs):
        super(Decoder, self).__init__(**kwargs)

    def init_state(self, enc_outputs, *args): # 用于将编码器的输出（enc_outputs）转换为编码后的状态，便于编码器使用
        raise NotImplementedError

    def forward(self, X, state): #  解码器在每个时间步都会将输入x （例如：在前一时间步生成的词元）和编码后的状态state 映射成当前时间步的输出词元
        raise NotImplementedError
class EncoderDecoder(nn.Module):
    """编码器-解码器架构的基类"""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args): # enc_X编码器输入，dec_X解码器输入
        enc_outputs = self.encoder(enc_X, *args) # enc_outputs编码器输出
        dec_state = self.decoder.init_state(enc_outputs, *args) # 将编码器的输出（enc_outputs）转换为编码后的状态，便于编码器使用
        return self.decoder(dec_X, dec_state)