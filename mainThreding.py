import grpc
from concurrent import futures
import threading
import numpy as np
import xgboost as xgb
from federated_pb2 import ModelWeights, Empty
from federated_pb2_grpc import FederatedLearningServicer, add_FederatedLearningServicer_to_server
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

# Kreiranje ML modela (XGBoost)
def create_ml_model():
    return xgb.XGBRegressor(n_estimators=100, random_state=42)

# Globalne liste za praćenje metrika
client_counts = []
r2_scores = []
mse_values = []
mae_values = []

class FederatedLearningServicer(FederatedLearningServicer):
    def __init__(self):
        self.global_model = create_ml_model()
        self.client_weights = []
        self.lock = threading.Lock()
        self.client_count = 0

    def UpdateModel(self, request, context):
        received_weights = np.array(request.weights)
        with self.lock:
            self.client_weights.append(received_weights)
            self.client_count += 1
            self.aggregate_weights()
            self.evaluate_model_performance()
        return Empty()

    def aggregate_weights(self):
        if self.client_weights:
            # Agregacija modelskih težina
            new_weights = np.mean(self.client_weights, axis=0)
            self.global_model.set_params(**dict(zip(self.global_model.get_booster().feature_names, new_weights)))
            self.client_weights = []

    def evaluate_model_performance(self):
        # Generišemo neke lažne podatke za testiranje performansi
        X_test = np.random.rand(100, 5)  # Primer generisanih podataka
        y_test = np.random.rand(100)
        
        # Predikcija i procena modela
        y_pred = self.global_model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        
        # Čuvanje metrika u listu
        client_counts.append(self.client_count)
        r2_scores.append(r2)
        mse_values.append(mse)
        mae_values.append(mae)

        # Prikaz napretka u konzoli
        print(f"Broj klijenata: {self.client_count}, R2: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")
        
        # Ažuriraj grafikon
        update_plot()

def update_plot():
    # Prikaz grafikona u realnom vremenu
    plt.clf()  # Očisti prethodni grafikon (ako ne želiš da se slike preklapaju)
    plt.subplot(1, 3, 1)
    plt.plot(client_counts, r2_scores, marker='o', color='b', label="R2 Score")
    plt.xlabel("Broj klijenata")
    plt.ylabel("R2 Score")
    plt.legend()
    plt.title("R2 Score vs Broj klijenata")
    
    plt.subplot(1, 3, 2)
    plt.plot(client_counts, mse_values, marker='o', color='r', label="MSE")
    plt.xlabel("Broj klijenata")
    plt.ylabel("MSE")
    plt.legend()
    plt.title("MSE vs Broj klijenata")
    
    plt.subplot(1, 3, 3)
    plt.plot(client_counts, mae_values, marker='o', color='g', label="MAE")
    plt.xlabel("Broj klijenata")
    plt.ylabel("MAE")
    plt.legend()
    plt.title("MAE vs Broj klijenata")
    
    plt.tight_layout()
    plt.pause(0.1)

def serve():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    add_FederatedLearningServicer_to_server(FederatedLearningServicer(), server)
    server.add_insecure_port('[::]:50051')
    server.start()
    print("GRPC server pokrenut na portu 50051.")
    plt.ion()  # Aktivacija interaktivnog prikaza grafikona
    plt.show()  # Prikaz grafikona
    server.wait_for_termination()

if __name__ == '__main__':
    serve()
