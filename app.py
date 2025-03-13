import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim

# Use a sidebar to let the user choose the page.
page = st.sidebar.selectbox(
    "Choose a demo", ["Simple Linear Model", "Complex NN Model"])

if page == "Simple Linear Model":
    st.title("Simple Linear Model Demo")
    st.write(
        "Here is a pre-generated dataset and a simple straight-line model (called the AI Model).")

    # Generate a simple dataset: y = 3*x + 5 + noise
    np.random.seed(0)
    x = np.linspace(0, 10, 50)
    noise = np.random.normal(0, 2, size=x.shape)
    y = 3 * x + 5 + noise

    # Display the dataset
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="blue", label="Data Points")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Test Score")
    ax.set_title("Pre-generated Data")
    st.pyplot(fig)

    st.write("Adjust the parameters of the straight-line AI Model.")
    slope = st.number_input("Slope", value=2.0, step=0.1)
    intercept = st.number_input("Intercept", value=4.0, step=0.1)

    # Compute predictions with the chosen parameters
    y_pred = slope * x + intercept

    # Compute goodness of fit (R² score)
    r2 = r2_score(y, y_pred)
    st.write(f"Goodness of fit (R²): **{r2:.3f}**")

    # Plot the AI model line with the data
    fig2, ax2 = plt.subplots()
    ax2.scatter(x, y, color="blue", label="Data Points")
    ax2.plot(x, y_pred, color="orange", linewidth=2, label="AI Model")
    ax2.set_xlabel("Study Hours")
    ax2.set_ylabel("Test Score")
    ax2.set_title("Data with AI Model")
    ax2.legend()
    st.pyplot(fig2)

elif page == "Complex NN Model":
    st.title("Complex NN Model Demo")
    st.write("Here is a more complex dataset. You can build a Neural Network (NN) with a chosen number of layers. Click 'Step' to let the NN learn a little more.")

    # Create a complex dataset: a noisy sine curve
    np.random.seed(0)
    x_complex = np.linspace(-10, 10, 200)
    y_complex = np.sin(x_complex) + np.random.normal(0,
                                                     0.2, size=x_complex.shape)

    # Plot the complex data
    fig, ax = plt.subplots()
    ax.scatter(x_complex, y_complex, color="blue", label="Data Points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Complex Data (Noisy Sine Wave)")
    st.pyplot(fig)

    # Convert data to torch tensors for training
    X_tensor = torch.tensor(x_complex, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y_complex, dtype=torch.float32).view(-1, 1)

    # Allow the user to choose the number of hidden layers (1 to 10)
    num_layers = st.slider("Number of Hidden Layers",
                           min_value=1, max_value=10, value=3)

    # Create a simple NN model if not already in session_state or if layers changed.
    if ('nn_model' not in st.session_state) or (st.session_state.get("num_layers", None) != num_layers):
        # Define a simple fully connected NN with ReLU activation
        class SimpleNN(nn.Module):
            def __init__(self, num_hidden_layers):
                super(SimpleNN, self).__init__()
                layers = []
                input_dim = 1
                hidden_dim = 10
                # Build hidden layers
                for i in range(num_hidden_layers):
                    in_features = input_dim if i == 0 else hidden_dim
                    layers.append(nn.Linear(in_features, hidden_dim))
                    layers.append(nn.ReLU())
                # Output layer
                layers.append(nn.Linear(hidden_dim, 1))
                self.model = nn.Sequential(*layers)

            def forward(self, x):
                return self.model(x)
        st.session_state.nn_model = SimpleNN(num_layers)
        st.session_state.num_layers = num_layers
        st.session_state.optimizer = optim.Adam(
            st.session_state.nn_model.parameters(), lr=0.01)
        st.session_state.epoch = 0

    st.write(f"Current training epoch: {st.session_state.epoch}")

    # When the user clicks the "Step" button, perform one training step.
    if st.button("Step"):
        model = st.session_state.nn_model
        optimizer = st.session_state.optimizer
        criterion = nn.MSELoss()
        model.train()
        optimizer.zero_grad()
        y_pred = model(X_tensor)
        loss = criterion(y_pred, y_tensor)
        loss.backward()
        optimizer.step()
        st.session_state.epoch += 1
        st.success(f"Training step complete. Loss: {loss.item():.4f}")

    # Display the NN's prediction over the training data
    model = st.session_state.nn_model
    model.eval()
    with torch.no_grad():
        y_nn_pred = model(X_tensor).numpy().flatten()

    # Compute loss on the training data
    loss_val = np.mean((y_nn_pred - y_complex) ** 2)

    # Plot the data with the NN prediction line
    fig2, ax2 = plt.subplots()
    ax2.scatter(x_complex, y_complex, color="blue", label="Data Points")
    ax2.plot(x_complex, y_nn_pred, color="orange",
             linewidth=2, label="NN Prediction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("NN Prediction vs. Data")
    ax2.legend()
    st.pyplot(fig2)
    st.write(f"Current Mean Squared Error (MSE): **{loss_val:.4f}**")
