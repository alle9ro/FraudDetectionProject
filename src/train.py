import joblib
from .models import build_random_forest, build_neural_network

def train_random_forest(X_train, y_train, model_path='outputs/models/rf_model.pkl'):
    """Train and save the Random Forest model."""
    model = build_random_forest()
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    return model

def train_neural_network(X_train, y_train, input_dim, epochs=50, batch_size=32, 
                        model_path='outputs/models/nn_model.keras'):
    """Train and save the Neural Network model."""
    model = build_neural_network(input_dim)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)
    model.save(model_path)
    return model