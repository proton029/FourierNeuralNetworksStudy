import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Set random seeds
torch.manual_seed(42)
np.random.seed(42)

# ============================================
# Fourier Series Neural Network
# ============================================

class FourierSeriesNN(nn.Module):
    """
    A neural network that uses Fourier Series as its architecture.
    f(x) = a0 + Sum(an * cos(nx) + bn * sin(nx))
    """
    def __init__(self, num_frequencies=10):
        super(FourierSeriesNN, self).__init__()
        
        # Fourier coefficients (learnable parameters)
        self.a0 = nn.Parameter(torch.randn(1))  # DC component (bias)
        
        # Cosine coefficients
        self.an = nn.Parameter(torch.randn(num_frequencies))
        
        # Sine coefficients
        self.bn = nn.Parameter(torch.randn(num_frequencies))
        
        # Store number of frequencies
        self.num_frequencies = num_frequencies
        
    def forward(self, x):
        """
        x: Input tensor of shape (batch_size, 1)
        """
        # Initialize output with DC component
        output = self.a0
        
        # Add Fourier series terms
        for n in range(1, self.num_frequencies + 1):
            # Cosine term: an * cos(n*x)
            output = output + self.an[n-1] * torch.cos(n * x)
            
            # Sine term: bn * sin(n*x)
            output = output + self.bn[n-1] * torch.sin(n * x)
            
        return output

class AdvancedFourierNN(nn.Module):
    """
    f(x) = a0 + SUm(an * cos(wn*x + theta))
    """
    def __init__(self, num_terms=10):
        super(AdvancedFourierNN, self).__init__()
        
        # DC component
        self.a0 = nn.Parameter(torch.randn(1))
        
        # Amplitude, frequency, and phase for each term
        self.amplitudes = nn.Parameter(torch.randn(num_terms))
        self.frequencies = nn.Parameter(torch.randn(num_terms))
        self.phases = nn.Parameter(torch.randn(num_terms))
        
        self.num_terms = num_terms
        
    def forward(self, x):
        output = self.a0
        
        for i in range(self.num_terms):
            # Each term: amplitude * cos(frequency * x + phase)
            output = output + self.amplitudes[i] * torch.cos(
                self.frequencies[i] * x + self.phases[i]
            )
            
        return output

# ============================================
# Training Data
# ============================================

def generate_training_data(target_function, x_range=(-np.pi, np.pi), 
                          num_points=1000, noise_std=0.05):
    """
    Generate training data for function approximation
    """
    x = np.linspace(x_range[0], x_range[1], num_points)
    y = target_function(x)
    
    # Add noise
    y_noisy = y + noise_std * np.random.randn(num_points)
    
    return x, y, y_noisy

# Different target functions to approximate
def square_wave(x):
    """Square wave function"""
    return np.where(np.sin(x) > 0, 1.0, -1.0)

def sawtooth_wave(x):
    """Sawtooth wave function"""
    return (x / np.pi) % 2 - 1

def triangle_wave(x):
    """Triangle wave function"""
    return 2 * np.abs(2 * (x / (2*np.pi) - np.floor(x/(2*np.pi) + 0.5))) - 1

def smooth_function(x):
    """Smooth but non-periodic function"""
    return np.sin(x) + 0.5 * np.sin(2*x) + 0.25 * np.sin(3*x)

# ============================================
# 2. Training Functions
# ============================================

def train_fourier_network(model, train_loader, epochs=500, lr=0.01):
    """
    Train the Fourier series network
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    losses = []
    
    print(f"\nTraining {model.__class__.__name__}...")
    for epoch in range(epochs):
        epoch_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            y_pred = model(x_batch)
            loss = criterion(y_pred, y_batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        
        avg_loss = epoch_loss / len(train_loader)
        losses.append(avg_loss)
        
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
    
    return losses

def visualize_results(model, x_train, y_train, x_test, y_true, title="Fourier Series Approximation"):
    """
    Visualize the network's approximation
    """
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(x_test.reshape(-1, 1))
        y_pred = model(x_test_tensor).numpy()
    
    plt.figure(figsize=(12, 8))
    
    # Plot training data
    plt.subplot(2, 1, 1)
    plt.scatter(x_train, y_train, s=1, alpha=0.5, label='Training data (noisy)')
    plt.plot(x_test, y_true, 'g-', linewidth=2, label='True function')
    plt.plot(x_test, y_pred, 'r--', linewidth=2, label='Fourier Network Prediction')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot error
    plt.subplot(2, 1, 2)
    error = np.abs(y_true - y_pred.flatten())
    plt.plot(x_test, error, 'b-', linewidth=1)
    plt.fill_between(x_test, 0, error, alpha=0.3)
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title(f'Prediction Error (MSE: {np.mean(error**2):.6f})')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'fourier_nn_{title.replace(" ", "_").lower()}.png', dpi=100)
    plt.show()
    
    return error

def visualize_fourier_components(model, x_test):
    """
    Visualize individual Fourier components
    """
    model.eval()
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(x_test.reshape(-1, 1))
        
        # Get DC component - FIXED: convert to scalar properly
        dc_value = model.a0.detach().cpu().numpy()
        # Handle both 0-dim and 1-dim arrays
        if isinstance(dc_value, np.ndarray):
            if dc_value.size == 1:
                dc = float(dc_value.item())
            else:
                dc = float(dc_value[0])
        else:
            dc = float(dc_value)
        
        # Get individual Fourier terms
        components = []
        
        if hasattr(model, 'an'):  # Basic FourierSeriesNN
            for n in range(1, model.num_frequencies + 1):
                an = model.an[n-1].detach().cpu().numpy().item()
                bn = model.bn[n-1].detach().cpu().numpy().item()
                cos_term = an * np.cos(n * x_test)
                sin_term = bn * np.sin(n * x_test)
                components.append((n, cos_term, sin_term, an, bn))
        
        # Calculate total prediction
        y_pred = model(x_test_tensor).numpy().flatten()
        
        # Determine number of components to plot
        n_components = min(6, len(components))
        
        # Create figure with appropriate size
        fig_height = 2 * (n_components + 1)
        fig, axes = plt.subplots(n_components + 1, 1, figsize=(12, fig_height))
        
        # If only one subplot, make axes iterable
        if n_components == 0:
            axes = [axes]
        
        # total prediction
        axes[0].plot(x_test, y_pred, 'k-', linewidth=2)
        axes[0].axhline(y=dc, color='r', linestyle='--', label=f'DC: {dc:.3f}')
        axes[0].set_title('Total Prediction')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlabel('x')
        axes[0].set_ylabel('f(x)')
        
        # Plot individual components
        for i, (n, cos_term, sin_term, an, bn) in enumerate(components[:n_components]):
            axes[i+1].plot(x_test, cos_term, label=f'{an:.3f} × cos({n}x)', alpha=0.7)
            axes[i+1].plot(x_test, sin_term, label=f'{bn:.3f} × sin({n}x)', alpha=0.7)
            axes[i+1].set_title(f'Frequency n={n} (|a|={abs(an):.3f}, |b|={abs(bn):.3f})')
            axes[i+1].legend(fontsize=8)
            axes[i+1].grid(True, alpha=0.3)
            axes[i+1].set_xlabel('x')
            axes[i+1].set_ylabel('Component')
        
        plt.tight_layout()
        plt.savefig('fourier_components.png', dpi=100)
        plt.show()
        
        return components

# ============================================
# 3. Lets Compare with Standard ANN with RELU activation
# ============================================

class SimpleMLP(nn.Module):
    """
    Standard multi-layer perceptron for comparison
    """
    def __init__(self, hidden_size=64):
        super(SimpleMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    
    def forward(self, x):
        return self.net(x)

def compare_with_mlp(fourier_model, mlp_model, train_loader, x_test, y_true):
    """
    Compare Fourier network with standard MLP
    """
    print("\n" + "="*50)
    print("Comparing Fourier Network with Standard MLP")
    print("="*50)
    
    # Train MLP
    print("\nTraining MLP...")
    mlp_losses = train_fourier_network(mlp_model, train_loader, epochs=500, lr=0.01)
    
    # Get predictions
    fourier_model.eval()
    mlp_model.eval()
    
    with torch.no_grad():
        x_test_tensor = torch.FloatTensor(x_test.reshape(-1, 1))
        fourier_pred = fourier_model(x_test_tensor).numpy().flatten()
        mlp_pred = mlp_model(x_test_tensor).numpy().flatten()
    
    # Visualize comparison
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(x_test, y_true, 'g-', linewidth=2, label='True')
    plt.plot(x_test, fourier_pred, 'r--', linewidth=2, label='Fourier Network')
    plt.plot(x_test, mlp_pred, 'b--', linewidth=2, label='MLP')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.title('Model Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    fourier_error = np.abs(y_true - fourier_pred)
    mlp_error = np.abs(y_true - mlp_pred)
    plt.plot(x_test, fourier_error, 'r-', label=f'Fourier (MSE: {np.mean(fourier_error**2):.6f})')
    plt.plot(x_test, mlp_error, 'b-', label=f'MLP (MSE: {np.mean(mlp_error**2):.6f})')
    plt.xlabel('x')
    plt.ylabel('Absolute Error')
    plt.title('Compare Error')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('fourier_vs_mlp.png', dpi=100)
    plt.show()

# ============================================
# Program Entry 
# ============================================

def main():
    print("="*60)
    print("Fourier Series Neural Network - Function Approximation")
    print("="*60)
    
    # Choose a target function
    functions = {
        '1': ('Square Wave', square_wave),
        '2': ('Sawtooth Wave', sawtooth_wave),
        '3': ('Triangle Wave', triangle_wave),
        '4': ('Smooth Function', smooth_function)
    }
    
    print("\nSelect a function to approximate:")
    for key, (name, _) in functions.items():
        print(f"{key}. {name}")
    
    choice = input("\nEnter your choice (1-4): ").strip()
    func_name, target_func = functions.get(choice, ('Square Wave', square_wave))
    
    print(f"\nApproximating: {func_name}")
    
    # Generate data
    print("\nGenerating training data...")
    x_train, y_true, y_noisy = generate_training_data(
        target_func, 
        x_range=(-np.pi, np.pi),
        num_points=500,
        noise_std=0.05
    )
    
    # Prepare data loader
    x_tensor = torch.FloatTensor(x_train.reshape(-1, 1))
    y_tensor = torch.FloatTensor(y_noisy.reshape(-1, 1))
    dataset = TensorDataset(x_tensor, y_tensor)
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    # Create Fourier network
    print("\nCreating Fourier Series Neural Network...")
    fourier_model = FourierSeriesNN(num_frequencies=15)
    print(f"Number of parameters: {sum(p.numel() for p in fourier_model.parameters())}")
    
    # Train Fourier network
    fourier_losses = train_fourier_network(fourier_model, train_loader, epochs=500)
    
    # Visualize results
    x_test = np.linspace(-np.pi, np.pi, 1000)
    y_test_true = target_func(x_test)
    
    try:
        visualize_results(fourier_model, x_train, y_noisy, x_test, y_test_true, 
                         f"Fourier Series Approximation - {func_name}")
    except Exception as e:
        print(f"Error in visualization: {e}")
    
    # Visualize individual Fourier components
    print("\nVisualizing Fourier components...")
    try:
        components = visualize_fourier_components(fourier_model, x_test)
    except Exception as e:
        print(f"Error in component visualization: {e}")
    
    # Compare with MLP
    print("\n" + "="*50)
    compare = input("\nCompare with standard MLP? (y/n): ").strip().lower()
    if compare == 'y':
        try:
            mlp_model = SimpleMLP(hidden_size=64)
            compare_with_mlp(fourier_model, mlp_model, train_loader, x_test, y_test_true)
        except Exception as e:
            print(f"Error in MLP comparison: {e}")
    
    # Show learned coefficients
    print("\n" + "="*50)
    print("Learned Fourier Coefficients:")
    # FIXED: Convert to scalar properly
    a0_val = fourier_model.a0.detach().cpu().numpy()
    if isinstance(a0_val, np.ndarray):
        if a0_val.size == 1:
            a0_val = a0_val.item()
        else:
            a0_val = a0_val[0]
    print(f"a0 (DC): {a0_val:.4f}")
    
    print("\nFirst 8 Fourier terms:")
    for i in range(min(8, fourier_model.num_frequencies)):
        an_val = fourier_model.an[i].detach().cpu().numpy()
        bn_val = fourier_model.bn[i].detach().cpu().numpy()
        
        # Convert to scalars
        if isinstance(an_val, np.ndarray):
            an_val = an_val.item() if an_val.size == 1 else an_val[0]
        if isinstance(bn_val, np.ndarray):
            bn_val = bn_val.item() if bn_val.size == 1 else bn_val[0]
            
        print(f"n={i+1:2d}: an={an_val:8.4f}, bn={bn_val:8.4f}, |magnitude|={np.sqrt(an_val**2 + bn_val**2):.4f}")

if __name__ == "__main__":
    main()