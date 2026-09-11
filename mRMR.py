import pandas as pd
import numpy as np
import argparse
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
from itertools import combinations
import ipaddress

def compute_mutual_information(X, y):
    return mutual_info_classif(X, y, discrete_features=False)

def compute_feature_redundancy(X):
    num_features = X.shape[1]
    redundancy_matrix = np.zeros((num_features, num_features))
    for i, j in combinations(range(num_features), 2):
        redundancy_matrix[i, j] = mutual_info_regression(X.iloc[:, [i]], X.iloc[:, j])[0]
        redundancy_matrix[j, i] = redundancy_matrix[i, j]
    return redundancy_matrix

def select_features(X, y, num_features):
    num_all_features = X.shape[1]
    mutual_info = compute_mutual_information(X, y)
    redundancy_matrix = compute_feature_redundancy(X)
    
    selected_features = []
    remaining_features = list(range(num_all_features))
    feature_scores = {}
    
    while remaining_features and len(selected_features) < num_features:
        best_feature = None
        best_score = -np.inf
        
        for feature in remaining_features:
            relevance = mutual_info[feature]
            redundancy = np.mean([redundancy_matrix[feature, f] for f in selected_features]) if selected_features else 0
            score = relevance - redundancy
            
            if score > best_score:
                best_score = score
                best_feature = feature
        
        if best_feature is not None:
            selected_features.append(best_feature)
            remaining_features.remove(best_feature)
            feature_scores[X.columns[best_feature]] = best_score
    
    sorted_feature_scores = sorted(feature_scores.items(), key=lambda item: item[1], reverse=True)
    return X.columns[selected_features], sorted_feature_scores

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="mRMR Feature Selection")
    parser.add_argument("--input", type=str, required=True, help="Input CSV file path")
    parser.add_argument("--output", type=str, required=True, help="Output CSV file path")
    parser.add_argument("--num_features", type=int, default=100, help="Number of features to select")
    parser.add_argument("--drop_columns", type=str, default="file_name,label,ip", help="Columns to drop (comma-separated)")
    
    args = parser.parse_args()
    
    stats_df = pd.read_csv(args.input)
    stats_df = stats_df.dropna()
    stats_df['ip'] = [int(ipaddress.IPv4Address(ip)) for ip in stats_df['ip']]
    
    drop_cols = [col.strip() for col in args.drop_columns.split(",")]
    X = stats_df.drop(columns=drop_cols)
    y = stats_df["label"]
    
    selected_features, feature_scores_list = select_features(X, y, num_features=args.num_features)
    
    df = pd.DataFrame(feature_scores_list, columns=['names', 'scores'])
    df.to_csv(args.output, index=False)
