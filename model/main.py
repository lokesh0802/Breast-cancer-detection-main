import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
import pickle5 as pickle


def create_model(data): 
    X = data.drop(['diagnosis'], axis=1)
  
    # scale the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # perform clustering
    kmeans = KMeans(n_clusters=2, random_state=42)
    data['cluster'] = kmeans.fit_predict(X_scaled)
    
    return kmeans, scaler


def train_logistic_regression_for_cluster(data, cluster, scaler):
    cluster_data = data[data['cluster'] == cluster]

    X_cluster = cluster_data.drop(['diagnosis', 'cluster'], axis=1)
    y_cluster = cluster_data['diagnosis']
    
    # scale the cluster data using the original scaler
    X_cluster_scaled = scaler.transform(X_cluster)

    # train the logistic regression model
    model = LogisticRegression()
    model.fit(X_cluster_scaled, y_cluster)

    return model


def get_clean_data():
    data = pd.read_csv("data/data.csv")
    
    data = data.drop(['Unnamed: 32', 'id'], axis=1)
    
    data['diagnosis'] = data['diagnosis'].map({'M': 1, 'B': 0})
    
    return data


def main():
    data = get_clean_data()

    # Step 1: Clustering
    kmeans, scaler = create_model(data)

    # Save the clustering model and scaler
    with open('model/kmeans_model.pkl', 'wb') as f:
        pickle.dump(kmeans, f)
        
    with open('model/scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)

    # Step 2: Train Logistic Regression for each cluster
    models = {}
    for cluster in range(2):  # Assuming 2 clusters, adjust based on actual number
        model = train_logistic_regression_for_cluster(data, cluster, scaler)
        models[cluster] = model

        # Save the logistic regression models
        with open(f'model/logistic_regression_model_cluster_{cluster}.pkl', 'wb') as f:
            pickle.dump(model, f)


if __name__ == '__main__':
    main()
