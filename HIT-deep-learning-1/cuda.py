import torch

# 此程序用来查询是否可用gpu加速
print(torch.cuda.is_available())
print(torch.cuda.device_count())
deviceId = torch.cuda.current_device()
print(deviceId)
print(torch.cuda.get_device_name(deviceId))