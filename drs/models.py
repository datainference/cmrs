import torch 
import torch.nn as nn
import torch.nn.functional as F


class bilstm(nn.Module):
    def __init__(self, embedding_matrix, num_classes, num_layers=2, hidden_size: int = 512, batch_first: bool =True, bidirectional: bool=True, dropout: float = 0.3):
        super(bilstm, self).__init__()
        # Embedding Layer
        _, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), 
                                                        freeze=False)
        # Bi-LSTM Layer
        self.bi_lstm = nn.LSTM(input_size=embedding_dim, 
                                hidden_size=hidden_size, 
                                num_layers=num_layers,
                                batch_first=batch_first, 
                                bidirectional=bidirectional, 
                                dropout=dropout)
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        # Sigmoid Function
        self.sg = nn.Sigmoid()

    def forward(self, input_x):
        embedded_sentence = self.embedding(input_x)
        lstm_out, _ = self.bi_lstm(embedded_sentence)
        logit = self.fc(torch.mean(lstm_out, dim=1))
        scores = self.sg(logit) 

        return logit, scores

class Transformer(nn.Module): 
    def __init__(self, embedding_matrix, num_classes, num_heads=4, num_layers=3, max_len=1024, batch_first: bool =True, dropout: float = 0.3):
        super(Transformer, self).__init__()
        # Embedding Layer
        _, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), 
                                                        freeze=False)
        # Positional Encoding
        self.pos_encoding = PositionalEncoding(embedding_dim, 
                                                max_len=max_len)
        # Transformer Encoder
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, 
                                                        nhead=num_heads, 
                                                        batch_first=batch_first)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, 
                                                        num_layers=num_layers)
        # Fully Connected Layer
        self.fc = nn.Linear(embedding_dim, num_classes)
        # Dropout
        self.dropout = nn.Dropout(dropout)
        # Sigmoid Function
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        x = self.pos_encoding(x)
        x = self.transformer_encoder(x)
        x = self.dropout(x)
        x = x.mean(dim=1)  # Global Average Pooling
        logit = self.fc(x)
        scores = self.sg(logit)
        return logit, scores


class AttentionRNN(nn.Module):
    def __init__(self, embedding_matrix, num_classes, num_layers=2, hidden_size: int = 512, batch_first: bool =True, bidirectional: bool=True, dropout: float = 0.3):
        super(AttentionRNN, self).__init__()
        # Embedding Layer
        _, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        # Bi-LSTM Layer
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=batch_first, bidirectional=bidirectional, num_layers=num_layers, dropout=dropout)
        # Attention layer with tanh activation
        self.attn_fc = nn.Linear(hidden_size * 2, hidden_size)  # Transform hidden states
        self.attn_tanh = nn.Tanh()  # Activation
        self.attn_weight = nn.Linear(hidden_size, 1)  # Compute scores
        # Fully Connected Layer
        self.fc = nn.Linear(hidden_size * 2, num_classes)  # Fully connected output layer
        # Sigmoid Function
        self.sg = nn.Sigmoid()

    def forward(self, x):
        x = self.embedding(x)
        lstm_out, _ = self.lstm(x)  # (batch, seq_len, hidden_dim * 2)
        attn_scores = self.attn_tanh(self.attn_fc(lstm_out))  # (batch, seq_len, hidden_dim)
        attn_scores = self.attn_weight(attn_scores)  # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_scores, dim=1)  # Normalize with softmax
        context_vector = torch.mean(torch.matmul(attn_weights.transpose(1, 2), lstm_out), dim=1)
        logit = self.fc(context_vector)
        scores = self.sg(logit)
        return logit, scores
    
    
class MLPTextClassifier(nn.Module):
    def __init__(self, embedding_matrix, hidden_dim: list[int] = [512, 256], num_classes: int = 338, dropout: float =  0.3, batch_first: bool =True):
        super(MLPTextClassifier, self).__init__()
        # Embedding Layer 
        _, embedding_dim = embedding_matrix.shape
        self.embedding = nn.Embedding.from_pretrained(torch.tensor(embedding_matrix, dtype=torch.float32), freeze=False)
        
        # FC Layer 
        self.fc1 = nn.Linear(embedding_dim, hidden_dim[0])
        self.norm1 = nn.LayerNorm(hidden_dim[0])
        self.dropout = nn.Dropout(dropout)
        # FC Layer 2
        self.fc2 = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.norm2 = nn.LayerNorm(hidden_dim[1])
        # Output Layer  
        self.output = nn.Linear(hidden_dim[1], num_classes)
        self.sg = nn.Sigmoid()

    def forward(self, x):  # x shape: (batch_size, seq_len)
        x = self.embedding(x)  # shape: (batch_size, seq_len, embed_dim)
        x = x.mean(dim=1)      # average over seq_len → (batch_size, embed_dim)
        x = self.fc1(x)
        x = self.norm1(x)
        x = F.elu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.norm2(x)
        x = F.elu(x)
        x = self.dropout(x)
        logit = self.output(x)
        scores = self.sg(logit)
        
        return logit, scores
    
class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.device = args.device
        self.BCEWithLogitsLoss = nn.BCEWithLogitsLoss()
        self.adj_ddi_major = args.adj_ddi_major
        self.adj_ddi_moder = args.adj_ddi_moder
        self.alpha = args.alpha
        self.beta = args.beta

    def DDIrate_logit(self, logit, adj_ddi):
        score_ddi = nn.Sigmoid()(logit)
        score_ddi = score_ddi.t() @ score_ddi
        score_ddi =  score_ddi.mul(adj_ddi).sum()
        return score_ddi / adj_ddi.sum().item()

    def forward(self, logits, input):
        l_bce = self.BCEWithLogitsLoss(logits, input.float())
        l_major_ddi = self.alpha * self.DDIrate_logit(logits, self.adj_ddi_major)
        l_moder_ddi = self.beta * self.DDIrate_logit(logits, self.adj_ddi_moder)

        return l_bce + l_major_ddi + l_moder_ddi, l_bce, l_major_ddi, l_moder_ddi


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1024):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]