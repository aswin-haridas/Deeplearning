import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.data import Field, BucketIterator, TabularDataset
import nltk
from nltk.tokenize import word_tokenize
import random

# Set random seed for reproducibility
SEED = 42
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

# Tokenization function
def tokenize(text):
    return word_tokenize(text)

# Load dataset
TEXT = Field(tokenize=tokenize, lower=True, init_token='<sos>', eos_token='<eos>')
fields = [('input', TEXT), ('response', TEXT)]
dataset = TabularDataset(
    path='input.txt',
    format='tsv',
    fields=fields
)

# Split dataset
train_data, test_data = dataset.split(split_ratio=0.8)

# Build vocabulary
TEXT.build_vocab(train_data, min_freq=2)

# Define model
class Chatbot(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(input_dim, hidden_dim)
        self.transformer = nn.Transformer(
            d_model=hidden_dim,
            nhead=2,
            num_encoder_layers=n_layers,
            num_decoder_layers=n_layers,
            dim_feedforward=hidden_dim,
            dropout=dropout
        )
        self.fc_out = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, src, trg):
        embedded_src = self.embedding(src)
        embedded_trg = self.embedding(trg)
        output = self.transformer(embedded_src, embedded_trg)
        output = self.fc_out(output)
        return output

# Initialize model and optimizer
INPUT_DIM = len(TEXT.vocab)
OUTPUT_DIM = len(TEXT.vocab)
HIDDEN_DIM = 256
N_LAYERS = 2
DROPOUT = 0.5

model = Chatbot(INPUT_DIM, OUTPUT_DIM, HIDDEN_DIM, N_LAYERS, DROPOUT)
optimizer = optim.Adam(model.parameters())

# Define loss function
criterion = nn.CrossEntropyLoss()

# Training function
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    for batch in iterator:
        src = batch.input
        trg = batch.response
        optimizer.zero_grad()
        output = model(src, trg[:-1])
        output_dim = output.shape[-1]
        output = output.view(-1, output_dim)
        trg = trg[1:].view(-1)
        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Evaluation function
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    with torch.no_grad():
        for batch in iterator:
            src = batch.input
            trg = batch.response
            output = model(src, trg[:-1])
            output_dim = output.shape[-1]
            output = output.view(-1, output_dim)
            trg = trg[1:].view(-1)
            loss = criterion(output, trg)
            epoch_loss += loss.item()
    return epoch_loss / len(iterator)

# Train the model
N_EPOCHS = 10
for epoch in range(N_EPOCHS):
    train_loss = train(model, train_data, optimizer, criterion)
    test_loss = evaluate(model, test_data, criterion)
    print(f'Epoch: {epoch+1}, Train Loss: {train_loss:.3f}, Test Loss: {test_loss:.3f}')

# Chat function
def chat(model, sentence, max_length=50):
    model.eval()
    tokens = tokenize(sentence)
    src_indexes = [TEXT.vocab.stoi[token] for token in tokens]
    src_tensor = torch.LongTensor(src_indexes).unsqueeze(1)
    trg_indexes = [TEXT.vocab.stoi['<sos>']]
    for i in range(max_length):
        trg_tensor = torch.LongTensor(trg_indexes).unsqueeze(1)
        output = model(src_tensor, trg_tensor)
        pred_token = output.argmax(2)[-1, :].item()
        trg_indexes.append(pred_token)
        if pred_token == TEXT.vocab.stoi['<eos>']:
            break
    trg_tokens = [TEXT.vocab.itos[i] for i in trg_indexes]
    return ' '.join(trg_tokens[1:-1])

# Example usage
user_input = input("You: ")
while user_input.lower() != 'quit':
    response = chat(model, user_input)
    print("Bot:", response)
    user_input = input("You: ")
