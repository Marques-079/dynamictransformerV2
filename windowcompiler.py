import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import torch.optim as optim

# Loads in the data
df = pd.read_csv('AAPL_min_data.csv')
df['date_time'] = pd.to_datetime(df['date_time'])
df = df.set_index('date_time').sort_index()

# Feature columns
feature_cols = [
    'volume', 'volume_weighted', 'open_price', 'close', 'high', 'low',
    'number_trades', 'log_return', 'high_low_spread', 'close_open_change',
    'sma_5', 'sma_15', 'ema_5', 'ema_15', 'volatility_5', 'volatility_15',
    'bollinger_upper', 'bollinger_lower', 'bollinger_width', 'atr_14',
    'volume_change_pct', 'vwap', 'obv', 'rsi_14', 'momentum_5', 'momentum_15',
    'ema_12', 'ema_26', 'macd', 'macd_signal', 'roc_10', 'stoch_k', 'stoch_d',
    'williams_r', 'cci', 'mfi'
]
label_col = "price_change_pct"  # labeled data


class windowcompiler:
    def __init__(self, df, feature_cols, label_col):
        self.df = df
        self.feature_cols = feature_cols
        self.label_col = label_col

    def create_rolling_windows(self, df, feature_cols, label_col, window_size=10):
        #Converting our price_change_pct to binary values up or down (unfortunately we lose strength of singal. maybe we can get this out of logit probabilities?)

        df['price_change_pct'] = df['price_change_pct'].apply(lambda x: 1 if x > 0 else 0)
        df['price_change_pct'] = df['price_change_pct'] #Im not shifting here because our dataloder applies the shift automatically
        df = df.iloc[:-1]


        #print(f'Y IS HERE {df["price_change_pct"]}: X return is here {df[feature_cols].values[0]}')

        # Extract the feature matrix
        feature_data = df[feature_cols].values
        
        # Extract the label array
        label_data = df[label_col].values
        #print(f'Label data is here {label_data[10:30]}')

  
        # Uncomment the line below to actually scale the data:
        #feature_data_scaled = scaler.fit_transform(feature_data)

        # Currently no scaling (comment out the next line if you want to scale)
        feature_data = feature_data

        # Create a 0-1 time index for one window
        time_index = np.linspace(0, 1, window_size).reshape(-1, 1)

        X, y = [], []
        n = len(df)
        
        for i in range(n - window_size):
            # Grab the unscaled data for just this window
            window_chunk = feature_data[i : i + window_size]  
            
            # Create a *new* scaler each time (can be MinMaxScaler or something else)
            scaler = MinMaxScaler(feature_range=(0, 1))
            window_chunk_scaled = scaler.fit_transform(window_chunk)
            
            # Then proceed to append your time index, etc.
            window_with_time = np.hstack([window_chunk_scaled, time_index])
            X.append(window_with_time)
            y.append(label_data[i + window_size])

        
        X = np.array(X)
        y = np.array(y)

        #print(f'Y IS HERE {df["price_change_pct"]}: X return is here {df[feature_cols].values[0]}')
        print(f'Y IS HERE {y[:10]}')

        
        return X, y

    def pytorch_convert(self):
        # Use self.df, self.feature_cols, self.label_col
        window_size = 10
        X, y = self.create_rolling_windows(
            self.df, self.feature_cols, self.label_col, window_size=window_size
        )

        # 80/20 train/test split
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        #print("X_train shape:", X_train.shape)
        #print("y_train shape:", y_train.shape)
        #print("X_test shape:", X_test.shape)
        #print("y_test shape:", y_test.shape)

        #print(f'DEBUGING {y_test[0]}')

        # Convert NumPy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)

        #print(f' X_train is here: {X_train_tensor[0]}')
        print(f'y_train is here: {y_train_tensor[:10]}')
        assert torch.all((y_train_tensor == 0) | (y_train_tensor == 1)), "Error: y_train_tensor contains values other than 0 and 1!"


        # Create PyTorch datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        # DataLoader settings
        batch_size = 32
        shuffle = True  # Ensures batches are randomized

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        # Progress bar for loading batches
        print("Loading training data...")
        for batch_X, batch_y in tqdm(train_loader, desc="Training Batches", unit="batch"):
            pass  # Normally you'd process the batch here

        print("Loading testing data...")
        for batch_X, batch_y in tqdm(test_loader, desc="Testing Batches", unit="batch"):
            pass  # Normally you'd process the batch here

        return train_loader, test_loader
    

#transformer code begins here :O
#---------------------- RUNNING THE CODE BELOW ----------------------#

if __name__ == "__main__":
    wc = windowcompiler(df, feature_cols, label_col)
    train_loader, test_loader = wc.pytorch_convert()

    #for X_batch, y_batch in train_loader:
    #    print("\nExample training batch:")
    #   print("X_batch shape:", X_batch.shape)
    #    print("y_batch shape:", y_batch.shape)
    #    break

    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LEARNING_RATE = 0.001

    # Transformer model hyperparameters
    D_MODEL = 128       
    NUM_HEADS = 4        
    NUM_LAYERS = 2      
    DROPOUT = 0.1        

    # Data dimensions 
    SEQ_LENGTH = 10     
    INPUT_DIM = 37  

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dropout):
        super(TransformerClassifier, self).__init__()
        # Embedding layer to project input features to d_model dimensions
        self.embedding = nn.Linear(input_dim, d_model)
        
        # Transformer encoder layers (using batch_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # ----- Added MLP (Feed-Forward Network) Block -----
        # This block scales up the representation to 2*d_model,
        # applies ReLU activation, dropout and LayerNorm,
        # then scales it back down to d_model.
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),   # scale up
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),   # scale down
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        
        # Final classification layer (1 output neuron for binary classification)
        self.fc = nn.Linear(d_model, 1)
        # Sigmoid activation to output probability between 0 and 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        # x: (batch_size, seq_length, input_dim)
        x = self.embedding(x)  # --> (batch_size, seq_length, d_model)
        x = self.transformer_encoder(x)  # --> (batch_size, seq_length, d_model)
        # Use the output of the last timestep as the aggregated representation
        x = x[:, -1, :]  # --> (batch_size, d_model)
        # ----- Pass through the extra MLP block -----
        x = self.mlp(x)  # --> (batch_size, d_model)
        x = self.fc(x)   # --> (batch_size, 1)
        x = self.sigmoid(x)  # --> (batch_size, 1) probability output
        return x
    

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TransformerClassifier(
    input_dim=INPUT_DIM,
    d_model=D_MODEL,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    dropout=DROPOUT
)
model.to(device)
print(device)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss for classification
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)



# ---------------- Training Loop ----------------
for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)  # Ensure y is (batch_size, 1)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)
    
    epoch_loss = running_loss / len(train_loader.dataset)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Training Loss: {epoch_loss:.4f}")
    
    # ---------------- Evaluation on Test Set ----------------
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
            outputs = model(batch_X)
            # Classify as 1 if output >= 0.5, else 0
            preds = (outputs >= 0.5).float()
            total += batch_y.size(0)
            correct += (preds == batch_y).sum().item()
    
    accuracy = correct / total
    print(f"Test Accuracy: {accuracy:.4f}")




#Maybe add time decay 
#More dropout
#MORE data x10 at least 