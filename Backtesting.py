# Make predictions
train_size = int(len(data) * 0.8)
test_size = len(data) - train_size
train_data, test_data = scaled_data[0:train_size, :], scaled_data[train_size:len(scaled_data), :]

X_test, y_test = create_sequences(test_data, time_step)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

predictions = model.predict(X_test)
predictions = scaler.inverse_transform(predictions)

# Plot predictions vs actual values
plt.figure(figsize=(16, 8))
plt.plot(data.index[train_size + time_step + 1:], scaler.inverse_transform(test_data[time_step:]), label='Actual Price')
plt.plot(data.index[train_size + time_step + 1:], predictions, label='Predicted Price')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

# Backtest strategy
def backtest_strategy(predictions, actual, threshold=0.01):
    buy_signals = []
    sell_signals = []
    for i in range(1, len(predictions)):
        if (predictions[i] - actual[i-1]) / actual[i-1] > threshold:
            buy_signals.append(i)
        elif (actual[i-1] - predictions[i]) / actual[i-1] > threshold:
            sell_signals.append(i)
    return buy_signals, sell_signals

buy_signals, sell_signals = backtest_strategy(predictions, scaler.inverse_transform(test_data[time_step:]))
