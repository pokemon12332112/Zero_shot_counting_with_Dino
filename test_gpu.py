import torch
#print(torch.cuda.get_device_properties(1).total_memory)
#devices = torch.get_all_devices()
devices = [d for d in range(torch.cuda.device_count())]
device_names  = [torch.cuda.get_device_name(d) for d in devices]
print(len(device_names))