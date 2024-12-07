import numpy as np
import pandas as pd
import streamlit as st

def oreste_with_row_sums(r_ij, r_j, alpha):
    # Initialize the d_ij matrix with the correct dimensions
    d_ij = np.zeros_like(r_ij, dtype=float)

    # Calculate d_ij for each element in the matrix
    for i in range(len(r_ij)):
        for j in range(len(r_j)):
            d_ij[i][j] = alpha * r_ij[i][j] + (1 - alpha) * r_j[j]
    
    # Flatten the matrix to get all the values in a 1D array
    flattened_matrix = d_ij.flatten()

    # Find the indices that would sort the flattened matrix in ascending order
    sorted_indices = np.argsort(flattened_matrix)

    # Sort the values in ascending order based on the indices
    sorted_values = flattened_matrix[sorted_indices]

    # Find the positions (indices) in the original matrix
    positions = [np.unravel_index(i, d_ij.shape) for i in sorted_indices]

    # Save the sorted values and their positions in a vector (list of tuples)
    sorted_vector = [(value, pos) for value, pos in zip(sorted_values, positions)]

    # Calculate the R vector
    R = []
    i = 0
    while i < len(sorted_vector):
        current_value = sorted_vector[i][0]
        sum_indices = 0
        count = 0
        
        # Collect all elements with the same value
        while i < len(sorted_vector) and sorted_vector[i][0] == current_value:
            position = i + 1  # Ranks start from 1
            sum_indices += position
            count += 1
            i += 1
        
        # Compute average rank for these values
        average_rank = sum_indices / count
        R.extend([average_rank] * count)

    # Create matrices for sorted_vector and R according to original positions
    sorted_matrix = np.zeros_like(d_ij, dtype=float)
    R_matrix = np.zeros_like(d_ij, dtype=float)

    for r_value, (value, pos) in zip(R, sorted_vector):
        sorted_matrix[pos] = value  # Populate sorted_vector values
        R_matrix[pos] = r_value  # Populate R values

    # Sum elements of R for the same row index (first coordinate) in sorted_vector
    row_sums = np.sum(R_matrix, axis=1)

    # Rank the row sums to find the best alternative based on lowest row sum
    ranked_alternatives = np.argsort(row_sums)  # Ascending order to get the best alternative with lowest row sum first

    return d_ij, sorted_matrix, R_matrix, row_sums, ranked_alternatives

# Streamlit interface
st.title("La méthode ORESTE")

# File upload
uploaded_file = st.file_uploader("Upload a CSV file (comma-separated)", type=["csv"])

if uploaded_file:
    try:
        # Read the CSV file into a DataFrame
        data = pd.read_csv(uploaded_file, header=None)

        # Extract r_j and r_ij from the DataFrame
        r_j = data.iloc[0].values  # First row is r_j
        r_ij = data.iloc[1:].values  # Remaining rows are r_ij

        # Display the extracted matrices
        st.subheader("Entrer Data")
        st.write("**Classement des critères selon Besson (r_j):**")
        st.write(r_j)

        st.write("**Classement des alternatives  selon Besson (r_ij):**")
        st.write(r_ij)

        # Alpha parameter
        alpha = st.slider("Alpha ", 0.0, 1.0, 0.5)

        # Call the function
        d_ij, sorted_matrix, R_matrix, row_sums, ranked_alternatives = oreste_with_row_sums(r_ij, r_j, alpha)

        # Display results
        st.subheader("Résultats ")
        st.write("**Matrice des distances projetées(d_ij):**")
        st.write(pd.DataFrame(d_ij))

        st.write("**Matrice de classement global de Besson basé sur les distances projetées:**")
        st.write(pd.DataFrame(R_matrix))

        st.write("**Classement global des alternatives:**")
        st.write(row_sums)

        # Rank the alternatives based on row sums and display the best one with lowest row sum
      
        for idx in ranked_alternatives:
            st.write(f"Alternative A{idx+1}: {row_sums[idx]}")

        # Display the best alternative (lowest row sum)
        best_alternative = ranked_alternatives[0]
        st.write(f"**Meilleure Alternative:** A{best_alternative+1}")

    except Exception as e:
        st.error(f"Error processing file: {e}")
else:
    st.info("Please upload a CSV file to proceed.")

