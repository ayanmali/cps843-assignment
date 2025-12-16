import torch
model = torch.load('output/crnn_word_classifier.pth', map_location=torch.device('cpu'))
print(model.keys())
print("=" * 60)
print("METADATA:")
print(model['metadata'])

# To print the model state dict, uncomment the following code
# print("=" * 60)
# print("MODEL STATE DICT:")
# print(model['model_state_dict'])