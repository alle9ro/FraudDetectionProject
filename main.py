from src.utils import ensure_dir
from src.data_preprocessing import load_data, preprocess_data
from src.train import train_random_forest, train_neural_network
from src.evaluate import evaluate_model

def main():
    ensure_dir('outputs/models')
    ensure_dir('outputs/plots')
    
    data_path = 'data/creditcard.csv'
    print("Loading dataset...")
    data = load_data(data_path)
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data)
    
    print("Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    
    print("Training Neural Network...")
    nn_model = train_neural_network(X_train, y_train, input_dim=X_train.shape[1])
    
    print("Evaluating Random Forest...")
    rf_report, rf_auc = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    print("Evaluating Neural Network...")
    nn_report, nn_auc = evaluate_model(nn_model, X_test, y_test, "Neural Network")
    
    print("\nSummary:")
    print(f"Random Forest AUC: {rf_auc:.2f}")
    print(f"Neural Network AUC: {nn_auc:.2f}")

if __name__ == "__main__":
    main()