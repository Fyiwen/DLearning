# 这里是计算l2损失的地方
def l2_loss(input, target, mask, batch_size):
    loss = (input - target) * mask
    loss = (loss * loss) / 2 / batch_size

    return loss.sum()
