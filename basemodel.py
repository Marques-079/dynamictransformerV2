import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import os

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

def load_minute_data_files(ticker_list):
    """
    Looks for each {ticker}_min_data.csv in the current working directory.
    """
    dfs = {}
    for i, ticker in enumerate(ticker_list, start=1):
        filename = f"{ticker}_min_data.csv"
        if os.path.exists(filename):
            df = pd.read_csv(filename)
            dfs[f"d{i}"] = df
            print(f"Loaded {filename} into variable d{i}")
        else:
            print(f"File {filename} not found in the local directory.")
    return dfs



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

            window_chunk = feature_data[i : i + window_size]  
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            window_chunk_scaled = scaler.fit_transform(window_chunk)
            
            window_with_time = np.hstack([window_chunk_scaled, time_index])
            X.append(window_with_time)
            y.append(label_data[i + window_size])

        
        X = np.array(X)
        y = np.array(y)

        #print(f'Y IS HERE {df["price_change_pct"]}: X return is here {df[feature_cols].values[0]}')
        #print(f'Y IS HERE {y[:10]}')
        #print(X.shape)
        #print(y.shape)

        
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




        num_zeros = np.count_nonzero(y_train == 0)
        num_ones = np.count_nonzero(y_train == 1)

        # Alternatively
        num_zeros = (y_train == 0).sum()
        num_ones = (y_train == 1).sum()

        print(f"NUMBER of zeros in y_train: {num_zeros}")
        print(f"NUMBER of ones in y_train: {num_ones}")


        # Convert NumPy arrays to PyTorch tensors
        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test, dtype=torch.float32)


        #print(f' X_train is here: {X_train_tensor[0]}')
        #print(f'y_train is here: {y_train_tensor[:10]}')
        assert torch.all((y_train_tensor == 0) | (y_train_tensor == 1)), "Error: y_train_tensor contains values other than 0 and 1!"


        # Create PyTorch datasets
        train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

        print("Train dataset size:", len(train_dataset))  # Should match num_samples in X_train_tensor
        print("Test dataset size:", len(test_dataset))  # Should match num_samples in X_test_tensor



        # DataLoader settings
        batch_size = 32
        shuffle = True  # Ensures batches are randomized

        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

        for batch_X, batch_y in train_loader:
            print("Batch X shape:", batch_X.shape)  # Should be (batch_size, sequence_length, num_features)
            print("Batch y shape:", batch_y.shape)  # Should be (batch_size, 1)
            break  # Only print first batch to avoid too much output


        # Progress bar for loading batches
        print("Loading training data...")
        for batch_X, batch_y in tqdm(train_loader, desc="Training Batches", unit="batch"):
            pass  # Normally you'd process the batch here

        print("Loading testing data...")
        for batch_X, batch_y in tqdm(test_loader, desc="Testing Batches", unit="batch"):
            pass  # Normally you'd process the batch here

        return train_loader, test_loader
    
#Compiler for different tickers TEST
    
def process_and_combine_datasets(dataframes_dict, feature_cols, label_col, batch_size=32, shuffle=True):
    train_datasets = []
    test_datasets = []

    # Process each dataset separately
    for key, df in dataframes_dict.items():
        wc = windowcompiler(df, feature_cols, label_col)
        train_loader, test_loader = wc.pytorch_convert()

        # Instead of converting DataLoader to a list of batches, collect the underlying datasets
        train_datasets.append(train_loader.dataset)
        test_datasets.append(test_loader.dataset)

    # Combine all datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_test_dataset = ConcatDataset(test_datasets)

    # Create final DataLoaders
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=shuffle)
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=shuffle)

    return combined_train_loader, combined_test_loader

    

#transformer code begins here :O
#---------------------- RUNNING THE CODE BELOW ----------------------#

if __name__ == "__main__":

    #-------------LOADS d[n] DATAFRAMES INTO A DICTIONARY-------------#

    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    LEARNING_RATE = 0.0001

    # Transformer model hyperparameters
    D_MODEL = 128  
    NUM_HEADS = 4       
    NUM_LAYERS = 2     
    DROPOUT = 0.3       

    # Data dimensions 
    SEQ_LENGTH = 10     
    INPUT_DIM = 37  


    tickers = ["AAPL", "GOOGL", "MSFT", "AMZN", "AMD", "CRM", "CRWD", "INTC", "INTU", "ORCL", "SHOP", "PLTR", "AVGO", "IBM", "SAP", "META", "TMUS", "T", "VZ", "QCOM", "UBER", "ADBE", "NOW",
              "ACN", "TXN", "ANET"]
    list_len = len(tickers)

    data_directory = "/Users/marcus/Documents/GitHub/dynamictransformerV2"

    dataframes_dict = load_minute_data_files(tickers)

    #df_first = dataframes_dict["d5"]  # pull the first DataFrame
    #print(df_first) 

    train_loader, test_loader = process_and_combine_datasets(
    dataframes_dict, feature_cols, label_col, batch_size=BATCH_SIZE
    )

    num_samples1 = len(train_loader)
    print("Total number of samples in train_loader:", num_samples1)

    num_samples2 = len(test_loader)
    print("Total number of samples in test_loader:", num_samples2)


class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, d_model, num_heads, num_layers, dropout):
        super(TransformerClassifier, self).__init__()
        # Embedding layer to project input features to d_model dimensions
        self.embedding = nn.Linear(input_dim, d_model)
        
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

        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_model * 2),   # scale up
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model),   # scale down
            nn.GELU(),
            nn.Dropout(dropout),
            nn.LayerNorm(d_model)
        )
        
        # Final classification layer (1 output neuron for binary classification)
        self.fc = nn.Linear(d_model, 1)
        # Sigmoid activation to output probability between 0 and 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
       
        x = self.embedding(x) 
        x = self.transformer_encoder(x)  
        
        x = x[:, -1, :]  
        # ----- Pass through the extra MLP block -----
        x = self.mlp(x)  
        x = self.fc(x)   
        x = self.sigmoid(x)  
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

optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
T_max = NUM_EPOCHS  # or a larger value if you want a slower decay
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=T_max,
    eta_min=1e-4
)

criterion = nn.BCELoss()  # Binary Cross Entropy Loss for classification





# ---------------- Training Loop ----------------


for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device).unsqueeze(1)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * batch_X.size(0)

    # Updates the scheduler after each epoch
    scheduler.step()

    epoch_loss = running_loss / len(train_loader.dataset)
    current_lr = scheduler.get_last_lr()[0]
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {epoch_loss:.4f}, LR: {current_lr:.8f}")
    
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

