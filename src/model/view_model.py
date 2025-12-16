import torch
model = torch.load('crnn_word_classifier.pth', map_location=torch.device('cpu'))
print(model['metadata'])