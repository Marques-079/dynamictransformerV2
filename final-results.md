# Evaluation of the $3 Transformer 💸

My goal was to create a transformer which can predict stocks from minute to minute (simulating intraday trading), the following report will run through the stages and thought process of the transformer model covering the entire project from preprocessing to model training & evaluation. The extra challenge here was doing it cost effectively without huge training costs and buying datasets.

```python
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader, ConcatDataset
from tqdm import tqdm
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import torch.nn as nn
import os
```

Above are the imported libraries for the main body of the transformer model

Of course our first goal is to retrieve data and then pre-process it before transformer training. I did this via a secondary script called **dataloader.py,** using Polygon.io’s free rest API (with a 1 minute cooldown per stock loading) we are able to extract the basic data minute to minute for a given ticker. This was the best API I was able to find on a **$0 budget.** The next step from here is data augmentation from our limited initial dataset of close, high, low, volume. Below is an example me extracting bollinger band features from this data 

```python
# Rolling Standard Deviation as a measure of volatility.
df['volatility_5'] = df['close'].rolling(window=5).std()
df['volatility_15'] = df['close'].rolling(window=15).std()

# Bollinger Bands (using the 15-period SMA and volatility)
df['bollinger_upper'] = df['sma_15'] + 2 * df['volatility_15']
df['bollinger_lower'] = df['sma_15'] - 2 * df['volatility_15']
df['bollinger_width'] = (df['bollinger_upper'] - df['bollinger_lower']) / df['sma_15']
```

After retrieving raw stock data, the next step is **feature selection and compilation** into structured DataFrames for each ticker. By extracting key technical indicators and financial metrics, we refine the dataset to ensure only the most relevant features are included. Each ticker's data is loaded into a dictionary containing the N tickers dataframe assigned to a specific key. 

```python
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
```

To ensure data quality, we performed multiple **checks for NaN values and class imbalances** in our dataset. Since the model learns from patterns in stock movements, it’s crucial that both `0s` and `1s` (representing up/down price movements) are evenly distributed. I also evaluated missing values to prevent bias in training. If imbalances or NaNs were detected, adjustments such as resampling or interpolation were applied. Below is a snapshot of the checks performed on a random dataframe before finalizing the dataset.

![Screenshot 2025-03-20 at 12.51.33 PM.png](https://github.com/Marques-079/dynamictransformerV2/blob/main/images/Osand1sfordynamictrans.jpg?raw=true)

![Screenshot 2025-03-20 at 12.20.08 PM.png](https://github.com/Marques-079/dynamictransformerV2/blob/main/images/ZerosandNaNs.jpg?raw=true)

---

We can see in image A that the numbers of `0s` vs `1s` is very similar but there is some difference, this can cause larger problems when scaling to massive datasets as this introduces a bias to the data and the transformer model. Overall we had a 1% difference between our `0s` and `1s` data. This should be noted.

```python
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
```

Since stock price predictions rely on sequential patterns, we transform the dataset into **time-based windows**, allowing the model to learn trends over multiple time steps. Each window captures a sequence of past stock movements and associated indicators, forming the basis for our predictions. These windows are then converted into **PyTorch tensors** and grouped into efficient **batches** for training. Below is an example of the transformation process.

```python
        for i in range(n - window_size):

            window_chunk = feature_data[i : i + window_size]  
            
            scaler = MinMaxScaler(feature_range=(0, 1))
            window_chunk_scaled = scaler.fit_transform(window_chunk)
            
            window_with_time = np.hstack([window_chunk_scaled, time_index])
            X.append(window_with_time)
            y.append(label_data[i + window_size])
```

Furthermore I was interested to find whether intra-window scaling or entire dataset scaling would make a difference. PS: It made a slight difference but negligible due to our TINY model size.

```python
def process_and_combine_datasets(dataframes_dict, feature_cols, label_col, batch_size=32, shuffle=True):
    train_datasets = []
    test_datasets = []

    for key, df in dataframes_dict.items():
        wc = windowcompiler(df, feature_cols, label_col)
        train_loader, test_loader = wc.pytorch_convert()

        train_datasets.append(train_loader.dataset)
        test_datasets.append(test_loader.dataset)

    # Combine all datasets
    combined_train_dataset = ConcatDataset(train_datasets)
    combined_test_dataset = ConcatDataset(test_datasets)

    # Create final DataLoaders
    combined_train_loader = DataLoader(combined_train_dataset, batch_size=batch_size, shuffle=shuffle)
    combined_test_loader = DataLoader(combined_test_dataset, batch_size=batch_size, shuffle=shuffle)

    return combined_train_loader, combined_test_loader
```

To ensure a robust evaluation of model performance, we split the compiled dataset into **training and testing sets**. The training set is used to optimize the model, while the test set evaluates its generalization ability. The test split occurs 80/20 to Training and Testing data respectively      (Over 1.3 million time steps I was able to scrap together)

# The Heart of our model [**❤️**](https://emojipedia.org/red-heart)

The following is the transformer model training and architecture, I have left out some parts to avoid this being 50 pages long but here is the rough idea:

```python
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
```

To effectively process stock data sequences, our model first applies an **embedding layer** that converts numerical features into a dense vector representation. This allows the model to better capture relationships between different features. The **transformer encoder** then processes these embeddings using attention mechanisms, enabling the model to learn temporal dependencies across multiple timesteps. Finally, before classification, the data is passed through an **MLP (Multi-Layer Perceptron) block**, which refines the learned representations before making a prediction. The Hyperparameters I used are mentioned later on :)

```python
    def forward(self, x):
       
        x = self.embedding(x) 
        x = self.transformer_encoder(x)  
        
        x = x[:, -1, :]  
        # ----- Pass through the extra MLP block -----
        x = self.mlp(x)  
        x = self.fc(x)   
        x = self.sigmoid(x)  
        return x
    
```

Small but powerful… Otherwise known as the multilayer perceptron this small function makes a HUGE difference introducing non-linearity into our data allowing it to capture far more complex relationships. We have also add a sigmoid classifier which converts our logits (prediction probabilities) into a binary 0 - 1 prediction, conveniently the same form as our labelled data.

![Screenshot 2025-03-20 at 12.32.41 PM.png](https://github.com/Marques-079/dynamictransformerV2/blob/main/images/1938293.jpg?raw=true)

Once the architecture is in place, the model undergoes **training** using historical stock price movements. The dataset is split into training and validation sets, and the model iteratively learns by adjusting weights to minimize error. We employ the **AdamW optimizer**, which adapts learning rates dynamically, and a **binary cross-entropy loss function** to measure performance. Training is performed in mini-batches to ensure stable learning aka groups before an update, and we monitor loss reduction across epochs to track improvements.

![Screenshot 2025-03-20 at 12.33.02 PM.png](https://github.com/Marques-079/dynamictransformerV2/blob/main/images/evalevaleval.jpg?raw=true)

To assess how well the model generalizes, we evaluate it on a separate **test dataset** after training. Key metrics such as **accuracy, precision, recall, and F1-score** are used to measure predictive ability. The test results allow us to determine if the model effectively captures stock price movements or if additional improvements are needed. Below is an example of the model’s performance evaluation.

---

![image.png](https://github.com/Marques-079/dynamictransformerV2/blob/main/images/lastepcohs21.png?raw=true)

---

The model demonstrates steady learning, stabilizing at **55.04% accuracy** by **epoch 20**, showing that it captures some predictive patterns in the data. The **gradual decline in loss** and adjustments in the learning rate indicate that the training process is functioning as expected, though the accuracy suggests room for further refinement. Given the relatively low epoch count, extending training could allow the model to reach a more optimal state. Additionally, improvements in **feature selection, data balancing, and hyperparameter tuning** could enhance performance, given extra compute power.

# A few more features 😎

![image.png](https://github.com/Marques-079/dynamictransformerV2/blob/main/images/Cosinecurve.jpg?raw=true)

(Graph Courtesy if ChatGPT)

![image.png](https://github.com/Marques-079/dynamictransformerV2/blob/main/images/Warmupcosine.jpg?raw=true)

Cosine Learning Rate Decay & Warmup

To prevent the model from plateauing too early, we use **cosine learning rate decay**, which gradually reduces the learning rate over time following a cosine curve. This allows the model to make **large updates** in the early stages of training while **fine-tuning with smaller adjustments** toward the later epochs. This technique helps improve convergence and prevents overshooting optimal weights. Below is a visualization of how the learning rate evolves during training.

---

As you can see and probably could have inferred from my code, even calling this model tiny would be an understatement. This took about 2 hours to train on an A100 GPU. I have intentionally included high dropout to prevent overfitting due to the small number of layers and heads. I hypothesise that the model is possibly underfitting to the data because of this.

```python
  # Transformer model hyperparameters
    BATCH_SIZE = 32
    NUM_EPOCHS = 20
    D_MODEL = 128  
    NUM_HEADS = 4       
    NUM_LAYERS = 2     
    DROPOUT = 0.3       

    # Data dimensions 
    SEQ_LENGTH = 10     
    INPUT_DIM = 37  
```

---

Overall this project (V2) is a much cleaner and tidy version of my past transformer projects. Huge thanks to **Andrej Karpathy** for inspiration on this project and an amazing video on Transformer model architecture. This was a valuable learning experience for me and by utilising libraries of pre-built components it aided my development greatly (instead of fully hand-rolled).

Although the results is arguably just over the 50% guessing mark there is much to improve on, and if I do decide to reattempt this project it will be with a much bigger budget.

If you made it to here, Thanks for reading :)

---
