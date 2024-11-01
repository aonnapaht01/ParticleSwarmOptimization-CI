import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load and preprocess data
data = pd.read_excel("AirQualityUCI.xlsx", na_values=-200).dropna()
features = data.iloc[:, [3, 6, 8, 10, 11, 12, 13, 14]].values
targets = data.iloc[:, 5].values

# Normalize the features
normalized_features = (features - features.mean(axis=0)) / features.std(axis=0)

# Set the structure of the neural network
hidden_layer_structure = [20]

class SimpleMLP:
    def __init__(self, input_dim, hidden_layers, output_dim):
        layer_sizes = [input_dim] + hidden_layers + [output_dim]
        self.weights = [np.random.rand(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)]

    def predict(self, inputs):
        for weight in self.weights:
            inputs = np.maximum(0, np.dot(inputs, weight))
        return inputs

class ParticleSwarm:
    def __init__(self, input_dim, hidden_layers, output_dim):
        self.network = SimpleMLP(input_dim, hidden_layers, output_dim)
        self.position = [w.copy() for w in self.network.weights]
        self.velocity = [np.random.randn(*w.shape) * 0.1 for w in self.position]
        self.best_position = self.position
        self.best_score = float("inf")

def optimize_with_pso(X, y, num_particles=15, num_iterations=200):
    swarm = [ParticleSwarm(X.shape[1], hidden_layer_structure, 1) for _ in range(num_particles)]
    global_best_position, global_best_score = None, float("inf")

    for _ in range(num_iterations):
        for particle in swarm:
            predictions = particle.network.predict(X).flatten()
            error = np.mean(np.abs(y - predictions))

            if error < particle.best_score:
                particle.best_score = error
                particle.best_position = [w.copy() for w in particle.position]

            if error < global_best_score:
                global_best_score = error
                global_best_position = [w.copy() for w in particle.position]

            # Update velocities and positions
            for i in range(len(particle.position)):
                r1 = np.random.rand(*particle.position[i].shape)
                r2 = np.random.rand(*particle.position[i].shape)
                particle.velocity[i] = (
                    0.5 * particle.velocity[i] +
                    1.5 * r1 * (particle.best_position[i] - particle.position[i]) +
                    1.5 * r2 * (global_best_position[i] - particle.position[i])
                )
                particle.position[i] += particle.velocity[i]

    return global_best_position

def cross_validation(X, y, num_folds=10):
    fold_size = len(X) // num_folds
    mae_scores = []

    for fold in range(num_folds):
        test_indices = range(fold * fold_size, (fold + 1) * fold_size)
        train_indices = list(set(range(len(X))) - set(test_indices))

        X_train, y_train = X[train_indices], y[train_indices]
        X_test, y_test = X[test_indices], y[test_indices]

        optimal_weights = optimize_with_pso(X_train, y_train)
        model = SimpleMLP(X_train.shape[1], hidden_layer_structure, 1)
        model.weights = optimal_weights

        predictions = model.predict(X_test).flatten()
        mae = np.mean(np.abs(y_test - predictions))
        mae_scores.append(mae)

        print(f"Fold {fold + 1}/{num_folds} - MAE: {mae}")

    return mae_scores

# Execute cross-validation and display results
mae_results = cross_validation(normalized_features, targets)
mean_mae = np.mean(mae_results)
print(f"Average MAE: {mean_mae}")

# Plot MAE for each fold
plt.figure(figsize=(10, 5))
plt.plot(range(1, len(mae_results) + 1), mae_results, marker='x', color="red")
plt.title("MAE per Fold")
plt.xlabel("Fold Number")
plt.ylabel("MAE")
plt.grid()
plt.show()

# Train final model on entire dataset
best_weights_final = optimize_with_pso(normalized_features, targets)
final_model = SimpleMLP(normalized_features.shape[1], hidden_layer_structure, 1)
final_model.weights = best_weights_final
final_predictions = final_model.predict(normalized_features).flatten()

# Plot Actual vs Predicted
plt.figure(figsize=(12, 6))
plt.plot(targets, label="Actual", color="green", alpha=0.6)
plt.plot(final_predictions, label="Predicted", color="pink", alpha=0.6)

# Vertical lines to indicate fold boundaries
for fold in range(1, 10):
    plt.axvline(x=fold * (len(normalized_features) // 10), color='black', linestyle='--', linewidth=1)

plt.title("Actual vs Predicted Benzene Concentration")
plt.xlabel("Sample Index")
plt.ylabel("Concentration (µg/m³)")
plt.legend()
plt.grid()
plt.show()
