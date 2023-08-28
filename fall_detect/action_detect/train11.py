# coding=utf-8
from action_detect.net import *
from data11 import *
from torch.utils.tensorboard import SummaryWriter

# 训练openpose+LSTM
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#DEVICE="cpu"
#input_size = 224要改成骨骼图片的尺寸啦
input_size = 32
hidden_size = 64
num_layers = 2
output_size = 2
frames_per_clip = 2
step_between_clips = 1

summmaryWriter = SummaryWriter("./logs")
train_dataset = FallDetectionDataset(video_path='F:/bishe/fall_detect/data/Lei2/Coffee_room_01/Videos',
                                     txt_path='F:/bishe/fall_detect/data/Lei2/Coffee_room_01/Annotation_files',
                                     frames_per_clip=frames_per_clip,
                                     step_between_clips=step_between_clips,
                                     input_size=input_size)
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_dataset = FallDetectionDataset(video_path='F:/bishe/fall_detect/data/Lei2/Home_02/Videos',
                                     txt_path='F:/bishe/fall_detect/data/Lei2/Home_02/Annotation_files',
                                     frames_per_clip=frames_per_clip,
                                     step_between_clips=step_between_clips,
                                     input_size=input_size)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
model = LSTM(input_size * input_size * 1, hidden_size, num_layers, output_size).to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)

num_epochs = 1
for epoch in range(num_epochs):
    train_sum_loss = 0  # 总训练损失

    train_total = 0
    sum_score = torch.tensor(0, dtype=torch.float32)
    for batch_idx, (data, targets) in enumerate(train_loader):
        data = data.to(DEVICE)  # 输入x的形状(frames_per_clip, 1, 图像高度, 图像宽度)
        targets = torch.as_tensor(targets, dtype=torch.long)
        targets = targets.to(DEVICE)
        model.train()  # 表明现在在训练环境下进行
        outputs = model(data).float()

        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_sum_loss += loss.cpu().detach().item()  # 训练总损失

        predict_targs = torch.argmax(outputs, dim=1)  # 预测的标签是取预测值y的行最大数的下标序号，因为y是有两个数的向量

        sum_score += torch.eq(predict_targs, targets).float().sum().cpu().detach()

    train_avg_loss = train_sum_loss / len(train_dataset)  # 计算训练平均损失
    train_score = sum_score.item() / len(train_dataset)

    test_sum_loss = 0
    sum_score = torch.tensor(0, dtype=torch.float32)

    for batch_idx, (data, targets) in enumerate(test_loader):
        data = data.to(DEVICE)  # 输入x的形状(frames_per_clip, 3, 图像高度, 图像宽度)
        targets = torch.as_tensor(targets, dtype=torch.long)

        targets = targets.to(DEVICE)
        model.eval()  # 标明在测试环境下
        outputs = model(data).float()

        loss = criterion(outputs, targets)
        test_sum_loss += loss.cpu().detach().item()  # 训练总损失
        predict_targs = torch.argmax(outputs, dim=1)
        sum_score += torch.eq(predict_targs, targets).float().sum().cpu().detach()

    test_avg_loss = test_sum_loss / len(test_loader)  # 计算训练平均损失
    test_score = sum_score.item() / len(test_dataset)

    # tensorboard --logdir=F:\bishe\fall_detect\action_detect\logs
    summmaryWriter.add_scalars("loss", {"train_avg_loss": train_avg_loss, "test_avg_loss": test_avg_loss},
                               epoch)  # 可视化时这个变量的名字为loss，要存档其内容，到时候在tensorboar可以查看根据其数据形成的图像
    summmaryWriter.add_scalars('accuracy', {'train_accuracy':train_score,'test_accuracy': test_score}, epoch)
    print('Epoch {}, train_avg_loss: {:.4f},test_avg_loss: {:.4f}'.format(epoch, train_avg_loss, test_avg_loss))  # 输出每一轮的平均训练和测试损失
    print('Epoch {}, train_avg_acc:{:.4f},test_avg_acc:{:.4f}'.format(epoch , train_score,test_score))
    torch.save(model.state_dict(),
               f"F:/bishe/fall_detect/action_detect/checkPoint/train/LSTMaction.jit")  # 在这个路径下保存训练得到的模型的每一层参数
