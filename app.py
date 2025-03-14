import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
import torch
import torch.nn as nn
import torch.optim as optim

# Helper: convert a numpy array to a LaTeX bmatrix.
def to_latex_bmatrix(A):
    if A.ndim == 1:
        A = A.reshape(-1, 1)
    rows = [" & ".join(f"{elem:.2f}" for elem in row) for row in A]
    return r"\begin{bmatrix}" + r" \\ ".join(rows) + r"\end{bmatrix}"

# Function to draw a neural network diagram with weights and biases.
def draw_neural_net(ax, left, right, bottom, top, layer_sizes, weights_biases=None):
    n_layers = len(layer_sizes)
    v_spacing = (top - bottom) / float(max(layer_sizes))
    h_spacing = (right - left) / float(n_layers - 1)
    
    if weights_biases is not None:
        all_weights = np.concatenate([np.abs(W).flatten() for (W, b) in weights_biases])
        max_weight = np.max(all_weights) if all_weights.size > 0 else 1.0
    else:
        max_weight = 1.0

    neuron_positions = []
    for i, layer_size in enumerate(layer_sizes):
        layer_positions = []
        layer_top = v_spacing * (layer_size - 1) / 2.0 + (top + bottom) / 2.0
        for j in range(layer_size):
            x_pos = left + i * h_spacing
            y_pos = layer_top - j * v_spacing
            circle = plt.Circle((x_pos, y_pos), v_spacing/4, color='w', ec='k', zorder=4)
            ax.add_artist(circle)
            if i > 0 and weights_biases is not None:
                bias = weights_biases[i-1][1][j]
                ax.text(x_pos, y_pos, f"{bias:.2f}", fontsize=8,
                        ha="center", va="center", zorder=5)
            layer_positions.append((x_pos, y_pos))
        neuron_positions.append(layer_positions)
    
    for i in range(n_layers - 1):
        if weights_biases is not None:
            W = weights_biases[i][0]
        else:
            W = None
        for j, (x1, y1) in enumerate(neuron_positions[i]):
            for k, (x2, y2) in enumerate(neuron_positions[i+1]):
                if W is not None:
                    weight = W[k, j]
                    norm_weight = weight / max_weight
                    color = plt.cm.seismic((norm_weight + 1) / 2)
                    lw = 1 + 2 * abs(norm_weight)
                else:
                    color = 'k'
                    lw = 1
                line = plt.Line2D([x1, x2], [y1, y2], c=color, linewidth=lw)
                ax.add_artist(line)

# Helper to get an activation module from a string.
def get_activation(name):
    if name == "ReLU":
        return nn.ReLU()
    elif name == "Sigmoid":
        return nn.Sigmoid()
    elif name == "Tanh":
        return nn.Tanh()
    elif name == "LeakyReLU":
        return nn.LeakyReLU()
    else:
        return nn.ReLU()

# Sidebar: Choose which demo page to view.
page = st.sidebar.selectbox("Choose a demo", ["Simple Linear Model", "Complex NN Model"])

if page == "Simple Linear Model":
    st.title("Simple Linear Model Demo")
    st.write("This page shows a pre-generated dataset and a simple AI Model (a straight line).")
    
    np.random.seed(0)
    x = np.linspace(0, 10, 50)
    noise = np.random.normal(0, 2, size=x.shape)
    y = 3 * x + 5 + noise
    
    st.sidebar.header("Model Parameters")
    slope = st.sidebar.number_input("Slope", value=2.0, step=0.1)
    intercept = st.sidebar.number_input("Intercept", value=4.0, step=0.1)
    
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)
    
    fig, ax = plt.subplots()
    ax.scatter(x, y, color="blue", label="Data Points")
    ax.set_xlabel("Study Hours")
    ax.set_ylabel("Test Score")
    ax.set_title("Pre-generated Data")
    st.pyplot(fig)
    
    st.write(f"Goodness of fit (R²): **{r2:.3f}**")
    
    fig2, ax2 = plt.subplots()
    ax2.scatter(x, y, color="blue", label="Data Points")
    ax2.plot(x, y_pred, color="orange", linewidth=2, label="AI Model")
    ax2.set_xlabel("Study Hours")
    ax2.set_ylabel("Test Score")
    ax2.set_title("Data with AI Model")
    formula = f"$y = {slope:.2f}x + {intercept:.2f}$"
    ax2.text(0.05, 0.95, formula, transform=ax2.transAxes, fontsize=14,
             verticalalignment='top', bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7))
    ax2.legend()
    st.pyplot(fig2)

elif page == "Complex NN Model":
    st.title("Complex NN Model Demo")
    st.write("This page shows a complex dataset and lets you build a Neural Network (NN) with custom settings.")
    
    np.random.seed(0)
    x_complex = np.linspace(-10, 10, 200)
    y_complex = np.sin(x_complex) + np.random.normal(0, 0.2, size=x_complex.shape)
    
    fig, ax = plt.subplots()
    ax.scatter(x_complex, y_complex, color="blue", label="Data Points")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Complex Data (Noisy Sine Wave)")
    st.pyplot(fig)
    
    X_tensor = torch.tensor(x_complex, dtype=torch.float32).view(-1, 1)
    y_tensor = torch.tensor(y_complex, dtype=torch.float32).view(-1, 1)
    
    st.sidebar.header("Neural Network Settings")
    num_layers = st.sidebar.slider("Number of Hidden Layers", min_value=1, max_value=10, value=3)
    lr = st.sidebar.slider("Learning Rate", min_value=0.0001, max_value=0.1, value=0.01, step=0.0001, format="%.4f")
    activation_name = st.sidebar.selectbox("Activation Function", ["ReLU", "Sigmoid", "Tanh", "LeakyReLU"])
    
    hidden_dim = 5
    
    # Reinitialize model if number of layers or activation function changes.
    if ('nn_model' not in st.session_state) or (st.session_state.get("num_layers", None) != num_layers) or (st.session_state.get("activation", None) != activation_name):
        class SimpleNN(nn.Module):
            def __init__(self, num_hidden_layers, act_name):
                super(SimpleNN, self).__init__()
                layers = []
                input_dim = 1
                for i in range(num_hidden_layers):
                    in_features = input_dim if i == 0 else hidden_dim
                    layers.append(nn.Linear(in_features, hidden_dim))
                    layers.append(get_activation(act_name))
                layers.append(nn.Linear(hidden_dim, 1))
                self.model = nn.Sequential(*layers)
            def forward(self, x):
                return self.model(x)
        st.session_state.nn_model = SimpleNN(num_layers, activation_name)
        st.session_state.num_layers = num_layers
        st.session_state.activation = activation_name
        st.session_state.optimizer = optim.Adam(st.session_state.nn_model.parameters(), lr=lr)
        st.session_state.lr = lr
        st.session_state.epoch = 0
    else:
        if st.session_state.get("lr", None) != lr:
            for param_group in st.session_state.optimizer.param_groups:
                param_group['lr'] = lr
            st.session_state.lr = lr
    
    st.write(f"Current training epoch: {st.session_state.epoch}")
    
    if st.button("Step"):
        model = st.session_state.nn_model
        optimizer = st.session_state.optimizer
        criterion = nn.MSELoss()
        for _ in range(10):
            model.train()
            optimizer.zero_grad()
            y_pred = model(X_tensor)
            loss = criterion(y_pred, y_tensor)
            loss.backward()
            optimizer.step()
            st.session_state.epoch += 1
        st.success(f"Training for 10 epochs complete. Loss: {loss.item():.4f}")
    
    model = st.session_state.nn_model
    model.eval()
    with torch.no_grad():
        y_nn_pred = model(X_tensor).numpy().flatten()
    
    loss_val = np.mean((y_nn_pred - y_complex) ** 2)
    
    fig2, ax2 = plt.subplots()
    ax2.scatter(x_complex, y_complex, color="blue", label="Data Points")
    ax2.plot(x_complex, y_nn_pred, color="orange", linewidth=2, label="NN Prediction")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.set_title("NN Prediction vs. Data")
    ax2.legend()
    
    weights_biases = []
    for module in st.session_state.nn_model.model:
        if isinstance(module, nn.Linear):
            W = module.weight.detach().numpy()
            b = module.bias.detach().numpy()
            weights_biases.append((W, b))
    
    layer_sizes = [1] + [hidden_dim] * num_layers + [1]
    fig3, ax3 = plt.subplots(figsize=(8, 4))
    ax3.axis('off')
    draw_neural_net(ax3, left=0.1, right=0.9, bottom=0.1, top=0.9,
                    layer_sizes=layer_sizes, weights_biases=weights_biases)
    
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(fig2)
    with col2:
        st.pyplot(fig3)
    
    st.write(f"Current Mean Squared Error (MSE): **{loss_val:.4f}**")
    
    # ---- Build the one-line model formula in LaTeX using the current activation ----
    act_str = r"\mathrm{" + activation_name + "}"
    current_expr = "x"
    for i, (W, b) in enumerate(weights_biases):
        W_latex = to_latex_bmatrix(W)
        b_latex = to_latex_bmatrix(b)
        if i < len(weights_biases) - 1:
            current_expr = act_str + r"\Big(" + W_latex + r"\cdot " + current_expr + r" + " + b_latex + r"\Big)"
        else:
            current_expr = W_latex + r"\cdot " + current_expr + r" + " + b_latex
    formula_str = r"y = " + current_expr
    
    st.write("### Model Formula (using actual weights, biases, and activation)")
    st.latex(formula_str)
    
    # ---- Display the activation function graph below the model formula ----
    st.write("### Activation Function Graph")
    x_act = np.linspace(-5, 5, 200)
    if activation_name == "ReLU":
        y_act = np.maximum(0, x_act)
    elif activation_name == "Sigmoid":
        y_act = 1 / (1 + np.exp(-x_act))
    elif activation_name == "Tanh":
        y_act = np.tanh(x_act)
    elif activation_name == "LeakyReLU":
        y_act = np.where(x_act < 0, 0.01*x_act, x_act)
    else:
        y_act = np.maximum(0, x_act)
    fig_act, ax_act = plt.subplots()
    ax_act.plot(x_act, y_act, color="purple", linewidth=2)
    ax_act.set_xlabel("x")
    ax_act.set_ylabel(activation_name + "(x)")
    ax_act.set_title(f"{activation_name} Activation Function")
    st.pyplot(fig_act)
