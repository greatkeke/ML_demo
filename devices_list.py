import torch

def list_available_devices():
    devices = []
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            devices.append(torch.cuda.get_device_name(i))
    else:
        devices.append("CPU")
    return devices

if __name__ == "__main__":
    available_devices = list_available_devices()
    for device in available_devices:
        print(device)

