import torch
import sys
import matplotlib.pyplot as plt
import numpy as np
sys.path.append("src/Utils")
from Utils.preprocess import test_loader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model=torch.load('ViT_Model/model_weight_MNIST.pt')
model.eval()



predictions=[]

for data, labels in test_loader:
    

    data = data.float()
    data = data.to(device)
    plt.imshow(np.transpose(data[0].cpu().detach().numpy(), (1, 2, 0)))
    with torch.no_grad():
        model=model.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, dim=1)
        predicted=predicted.to('cpu')

    break

for data, labels in test_loader:
    

    data = data.float()
    data = data.to(device)
    plt.imshow(np.transpose(data[0].cpu().detach().numpy(), (1, 2, 0)))
    with torch.no_grad():
        model=model.to(device)
        outputs = model(data)
        _, predicted = torch.max(outputs, dim=1)
        predicted=predicted.to('cpu')
        predictions.append(predicted)

torch.save(predictions, 'ViT_Model/model_weight_MNIST.pt')