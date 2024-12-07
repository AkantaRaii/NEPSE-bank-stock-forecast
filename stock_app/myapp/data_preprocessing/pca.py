
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import os
from  django.conf import settings 
# Loop through each stock
def apply_pca(stock,data):
    # Load the dataset for each stock
    date=data['date']
    # Step 1: Data Standardization (excluding 'date', 'Y_close', and 'close')
    features = data.drop(columns=['date'])
    close_column = data['close']  # Extract 'close' column to add it back later
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(features)

    # Convert standardized data back to DataFrame for easier handling
    standardized_df = pd.DataFrame(standardized_data, columns=features.columns)

    # Step 2: Calculate the Covariance Matrix
    cov_matrix = np.cov(standardized_df.T)

    # Step 3: Calculate Eigenvalues and Eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

    # Step 4: Sort Eigenvalues and Select Principal Components
    # Create a list of (eigenvalue, eigenvector) tuples
    eigen_pairs = [(np.abs(eigenvalues[i]), eigenvectors[:, i]) for i in range(len(eigenvalues))]
    eigen_pairs.sort(key=lambda x: x[0], reverse=True)

    # Select the top k eigenvectors (for example, let's keep 2 principal components)
    k =5
    selected_eigenvectors = np.column_stack([eigen_pairs[i][1] for i in range(k)])

    # Step 5: Transform the Data
    pca_transformed_data = standardized_df.dot(selected_eigenvectors)
    pca_transformed_df = pd.DataFrame(np.array(pca_transformed_data), columns=[f'PC{i+1}' for i in range(k)])

    # Concatenate the PCA-transformed data with the 'date' column
    final_df = pd.concat([pca_transformed_df, date.reset_index(drop=True)], axis=1)

    # Ensure there are no complex numbers in the DataFrame
    final_df = final_df.apply(lambda col: col.map(lambda x: np.real(x) if np.iscomplexobj(x) else x))

    # Add the 'close_column' to the DataFrame
    final_df = pd.concat([final_df, close_column.reset_index(drop=True)], axis=1)

    # Save the final DataFrame to a CSV file
    final_df.to_csv("transformed_data.csv", index=False)




    final_df.to_csv(os.path.join(settings.BASE_DIR,'../','data','pca_data',f'p{stock}.csv'), index=False)
    return final_df
