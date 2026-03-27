"""
Lightweight Transformer Architecture for Daily Flood Hazard Classification Driven by MJO Dynamics.
This script encompasses data preprocessing, model training (Transformer, LSTM, RF),
comparative evaluation, Explainable AI (XAI) extraction, and real-time simulation.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

# ==========================================
# PHASE 1: DATA PREPROCESSING AND SEQUENCE GENERATION
# ==========================================

def categorize_rainfall(rf):
    """
    Converts daily rainfall (mm) into four-tiered emergency hazard levels.
    0: Safe (<5), 1: Advisory (5-20), 2: Watch (20-40), 3: Warning (>40)
    """
    if rf < 5: return 0
    elif rf <= 20: return 1
    elif rf <= 40: return 2
    else: return 3

# Load historical dataset (assumes CSV format)
# The observation period spans from January 1, 2016, to December 31, 2025.
data = pd.read_csv('Sumbawa_Daily_2016_2025.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Define F=5 input features: Rainfall, RMM1, RMM2, Phase, and Amplitude
features = data[['Curah Hujan', 'RMM1', 'RMM2', 'Phase', 'Amplitude']].values
targets = data['Curah Hujan'].apply(categorize_rainfall).values

# Normalize feature scales using Min-Max Scaling to a [0, 1] range
scaler = MinMaxScaler()
features_scaled = scaler.fit_transform(features)

def create_sequences(features, targets, dates, window_size=14):
    """
    Transforms the 2D matrix into 3D tensors using a sliding window approach.
    Constructs a tensor shape of (T=14, F=5) to preserve sequence integrity.
    """
    X, y, target_dates = [], [], []
    for i in range(len(features) - window_size):
        X.append(features[i:(i + window_size)])
        y.append(targets[i + window_size])
        target_dates.append(dates.iloc[i + window_size])
    return np.array(X), np.array(y), np.array(target_dates)

X_seq, y_seq, dates_seq = create_sequences(features_scaled, targets, data['Date'], window_size=14)

# Apply One-Hot Encoding to the categorical target labels
y_seq_categorical = tf.keras.utils.to_categorical(y_seq, num_classes=4)

# Execute chronological partitioning: 70% Training, 15% Validation, 15% Testing
# This strict split prevents temporal data leakage into the algorithmic training phase.
train_idx = int(len(X_seq) * 0.70)
val_idx = int(len(X_seq) * 0.85)

X_train, y_train = X_seq[:train_idx], y_seq_categorical[:train_idx]
X_val, y_val = X_seq[train_idx:val_idx], y_seq_categorical[train_idx:val_idx]
X_test, y_test, dates_test = X_seq[val_idx:], y_seq_categorical[val_idx:], dates_seq[val_idx:]

# ==========================================
# PHASE 2: LIGHTWEIGHT TRANSFORMER ARCHITECTURE
# ==========================================

def positional_encoding(length, depth):
    """
    Injects temporal sequence identity into the embedding matrix using 
    sine and cosine trigonometric functions.
    """
    depth = depth / 2
    positions = np.arange(length)[:, np.newaxis]
    depths = np.arange(depth)[np.newaxis, :] / depth
    angle_rates = 1 / (10000 ** depths)
    angle_rads = positions * angle_rates
    pos_encoding = np.concatenate([np.sin(angle_rads), np.cos(angle_rads)], axis=-1)
    return tf.cast(pos_encoding, dtype=tf.float32)

class TransformerBlock(layers.Layer):
    """
    Constructs the parallel processing core comprising Multi-Head Self-Attention 
    and a Feed-Forward Network (FFN).
    """
    def __init__(self, embed_dim=32, num_heads=4, ff_dim=64, rate=0.2):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation="relu"),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        # Dropout layer implemented to mitigate overfitting risk
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training=False):
        # Extract attention weights for physical interpretability
        attn_output, attn_weights = self.att(inputs, inputs, inputs, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output), attn_weights

# Initialize the Functional API Model
inputs = layers.Input(shape=(14, 5))
x = layers.Dense(32)(inputs)
x = x + positional_encoding(length=14, depth=32)

transformer_block = TransformerBlock(embed_dim=32, num_heads=4, ff_dim=64, rate=0.2)
x, attention_weights = transformer_block(x)

# Classification Layer mapped to 4 categorical hazard probabilities
x = layers.GlobalAveragePooling1D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(4, activation="softmax")(x)

model_tf = models.Model(inputs=inputs, outputs=[outputs, attention_weights])

# ==========================================
# PHASE 3: TRAINING STRATEGY AND OPTIMIZATION
# ==========================================

# Compile the model using Adam optimizer and Categorical Cross-Entropy loss
model_tf.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), 
    loss=['categorical_crossentropy', None], 
    metrics=['accuracy']
)

# Implement Early Stopping to halt training autonomously upon convergence
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_dense_1_loss', 
    patience=15, 
    restore_best_weights=True
)

print("\n--- Initiating Lightweight Transformer Training ---")
model_tf.fit(
    X_train, y_train, 
    validation_data=(X_val, y_val), 
    epochs=80, 
    batch_size=32, 
    callbacks=[early_stopping], 
    verbose=0
)

# ==========================================
# PHASE 4: BASELINE MODELS EVALUATION (LSTM & RANDOM FOREST)
# ==========================================

print("\n--- Training LSTM Baseline Model ---")
lstm_inputs = layers.Input(shape=(14, 5))
h = layers.LSTM(64)(lstm_inputs) # 64 hidden memory units
h = layers.Dropout(0.2)(h)
lstm_outputs = layers.Dense(4, activation="softmax")(h)

model_lstm = models.Model(inputs=lstm_inputs, outputs=lstm_outputs)
model_lstm.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model_lstm.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=80, batch_size=32, callbacks=[early_stopping], verbose=0)

print("\n--- Training Random Forest Baseline Model ---")
# Flatten the 3D tensor (14, 5) into a 1D vector (70) as tree algorithms lack native sequential processing
X_train_flat = X_train.reshape((X_train.shape[0], -1))
X_test_flat = X_test.reshape((X_test.shape[0], -1))

rf_model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
rf_model.fit(X_train_flat, np.argmax(y_train, axis=1))

# ==========================================
# PHASE 5: COMPARATIVE PERFORMANCE ANALYSIS
# ==========================================

y_true_classes = np.argmax(y_test, axis=1)
target_labels = ['Safe', 'Advisory', 'Watch', 'Warning']

# Transformer Evaluation
pred_tf, extracted_att = model_tf.predict(X_test)
print("\n[Proposed Lightweight Transformer Performance]")
print(classification_report(y_true_classes, np.argmax(pred_tf, axis=1), target_names=target_labels))

# LSTM Evaluation
print("\n[LSTM Baseline Performance]")
print(classification_report(y_true_classes, np.argmax(model_lstm.predict(X_test), axis=1), target_names=target_labels))

# Random Forest Evaluation
print("\n[Random Forest Baseline Performance]")
print(classification_report(y_true_classes, rf_model.predict(X_test_flat), target_names=target_labels))

# ==========================================
# PHASE 6: EXPLAINABLE AI (XAI) - GEOSCIENCE INTERPRETABILITY
# ==========================================

print("\n--- XAI Extraction: Attention Weights across MJO Phases ---")
# Retrieve the actual MJO phase from the original unscaled test set
phase_index = 3 
actual_phases_test = scaler.inverse_transform(X_test[:, -1, :])[:, phase_index]

# Aggregate attention weights by averaging across all four attention heads
mean_att_per_seq = np.mean(extracted_att, axis=(1, 2, 3)) 

# Map the mean attention weights to the eight distinct MJO phases
phase_attention_map = {int(p): [] for p in range(1, 9)}
for seq_idx, phase in enumerate(actual_phases_test):
    phase = int(np.round(phase))
    if 1 <= phase <= 8:
        phase_attention_map[phase].append(mean_att_per_seq[seq_idx])

for p in range(1, 9):
    avg_att = np.mean(phase_attention_map[p]) if phase_attention_map[p] else 0
    print(f"MJO Phase {p}: Average Attention Weight = {avg_att:.4f}")

# ==========================================
# PHASE 7: TACTICAL PREDICTIVE MAPPING (JANUARY 2025 SIMULATION)
# ==========================================

print("\n--- Real-Time Early Warning Simulation: January 2025 ---")
# Extract a specific simulation window from the chronological test subset
start_date = pd.to_datetime('2025-01-10')
end_date = pd.to_datetime('2025-01-19')

mask = (pd.to_datetime(dates_test) >= start_date) & (pd.to_datetime(dates_test) <= end_date)
X_sim = X_test[mask]
dates_sim = dates_test[mask]

if len(X_sim) > 0:
    pred_sim, _ = model_tf.predict(X_sim)
    print(f"{'Date':<15} | {'P(Safe)':<10} | {'P(Advisory)':<12} | {'P(Watch)':<10} | {'P(Warning)':<10}")
    print("-" * 70)
    for i in range(len(dates_sim)):
        date_str = str(dates_sim[i].date())
        p_safe, p_adv, p_watch, p_warn = pred_sim[i]
        print(f"{date_str:<15} | {p_safe*100:5.2f}%    | {p_adv*100:5.2f}%      | {p_watch*100:5.2f}%    | {p_warn*100:5.2f}%")
else:
    print("The specified date range for January 2025 is not found within the current test subset.")
