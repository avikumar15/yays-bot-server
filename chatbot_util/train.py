import json
import numpy as np

from chat_dataset import ChatDataset
from model import NeuralNetwork
from nltk_wrapper import tokenise, stem, bag_of_words

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

with open('train.json', 'r') as f:
    data = json.load(f)

print(data)

words = []
classifications = []
collections = []

for item in data['classes']:
    classifications += [item['tag']]
    for pattern in item['patterns']:
        tokenised_pattern = tokenise(pattern)
        words.extend(tokenised_pattern)
        collections += [(tokenised_pattern, item['tag'])]

ignore_words = ['.', ',', '?', '!', '<', '>']
words = [stem(w) for w in words if w not in ignore_words]
words = sorted(set(words))
print(words)

x_train = []
y_train = []

for tup in collections:
    bag_words = bag_of_words(tup[0], words)
    x_train += [bag_words]
    y_train += [classifications.index(tup[1])]

x_train = np.array(x_train)
y_train = np.array(y_train)

print(x_train, y_train)

batch_size = 8

input_size = len(words)
hidden_size = 8
output_size = len(classifications)

learning_rate = 0.001
num_epochs = 1000

print(input_size, len(x_train[0]))
print(output_size, classifications)

dataset = ChatDataset(x_train=x_train, y_train=y_train)
train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = NeuralNetwork(input_size=input_size, hidden_size=hidden_size, num_classes=output_size).to(device)

# Loss and Optimiser
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

print(device)

for epoch in range(num_epochs):
    for (words, labels) in train_loader:
        words = words.to(dtype=torch.float).to(device)
        labels = labels.to(dtype=torch.long).to(device)

        # Forward pass
        outputs = model(words)
        # if y would be one-hot, we must apply
        # labels = torch.max(labels, 1)[1]
        loss = criterion(outputs, labels)

        # Backward and optimize
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

print(f'Final loss={loss.item():.4f}')

data = {
    "model_state": model.state_dict(),
    "input_size": input_size,
    "hidden_size": hidden_size,
    "output_size": output_size,
    "all_words": words,
    "tags": classifications
}

FILE = "data.pth"
torch.save(data, FILE)

print(f'training complete. file saved to {FILE}')
