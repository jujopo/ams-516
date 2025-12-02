import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from tqdm import tqdm

# ====================
# Model Parameters (Baseline Scenario S1)
# ====================
class Params:
    # Model parameters
    mu = 0.2       # drift of stock P
    sigma = 0.4    # volatility of stock P
    theta = 0.1    # long-term equilibrium level of spread
    kappa = 1.0    # speed of mean reversion
    nu = 0.15      # volatility of spread
    rho = 0.5      # correlation between stock P and spread
    r = 0.01       # risk-free rate
    gamma = 5.0    # risk aversion (CARA parameter)
    T = 1.0        # time horizon
    
    # Transaction cost parameters
    zeta_p = 0.0005  # proportional cost for buying P
    eta_p = 0.0005   # proportional cost for selling P  
    zeta_q = 0.0005  # proportional cost for buying Q
    eta_q = 0.0005   # proportional cost for selling Q
    
    # Derived parameters
    a_p = 1 + zeta_p
    b_p = 1 - eta_p
    a_q = 1 + zeta_q
    b_q = 1 - eta_q
    
    # Domain bounds
    t_min, t_max = 0.0, T
    p_min, p_max = 0.3, 4.0
    x_min, x_max = -0.5, 0.5
    y_min, y_max = -20.0, 20.0

params = Params()

# ====================
# Neural Network Architecture
# ====================
class PINN(nn.Module):
    def __init__(self, n_layers=6, n_neurons=64):
        super(PINN, self).__init__()
        
        # Network for u(t, p, x, y) = log H(t, p, x, y)
        self.u_net = self._build_net(4, 1, n_layers, n_neurons)
        
        # Network for buy boundary Y_b(t, p, x)
        self.Yb_net = self._build_net(3, 1, n_layers, n_neurons)
        
        # Network for sell boundary Y_s(t, p, x)
        self.Ys_net = self._build_net(3, 1, n_layers, n_neurons)
        
    def _build_net(self, input_dim, output_dim, n_layers, n_neurons):
        layers = []
        layers.append(nn.Linear(input_dim, n_neurons))
        layers.append(nn.Tanh())
        
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(n_neurons, n_neurons))
            layers.append(nn.Tanh())
            
        layers.append(nn.Linear(n_neurons, output_dim))
        return nn.Sequential(*layers)
    
    def forward(self, t, p, x, y):
        # Ensure all inputs have shape (batch_size, 1)
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        if p.dim() == 1:
            p = p.unsqueeze(-1)
        if x.dim() == 1:
            x = x.unsqueeze(-1)
        if y.dim() == 1:
            y = y.unsqueeze(-1)
        
        # Concatenate inputs
        u_input = torch.cat([t, p, x, y], dim=1)
        u = self.u_net(u_input)
        
        # Boundary networks
        boundary_input = torch.cat([t, p, x], dim=1)
        Yb = self.Yb_net(boundary_input)
        Ys = self.Ys_net(boundary_input)
        
        return u, Yb, Ys

# ====================
# Helper Functions
# ====================
def sample_domain(batch_size, device, dtype=torch.float64):
    """Sample collocation points uniformly from the domain"""
    t = torch.rand(batch_size, 1, device=device, dtype=dtype) * params.T
    p = torch.rand(batch_size, 1, device=device, dtype=dtype) * (params.p_max - params.p_min) + params.p_min
    x = torch.rand(batch_size, 1, device=device, dtype=dtype) * (params.x_max - params.x_min) + params.x_min
    y = torch.rand(batch_size, 1, device=device, dtype=dtype) * (params.y_max - params.y_min) + params.y_min
    return t, p, x, y

def A_plus(p, x):
    """A_+(p,x) = (b_p - a_q * e^x) * p"""
    return (params.b_p - params.a_q * torch.exp(x)) * p

def A_minus(p, x):
    """A_-(p,x) = (a_p - b_q * e^x) * p"""
    return (params.a_p - params.b_q * torch.exp(x)) * p

def terminal_condition(p, x, y):
    """Terminal condition: u(T,p,x,y) = -γ * J(p,x,y)"""
    J = torch.where(y >= 0, 
                    A_plus(p, x) * y,
                    A_minus(p, x) * y)
    return -params.gamma * J

# ====================
# Physics-Informed Loss Functions
# ====================
def compute_losses(model, t, p, x, y, device='cpu'):
    """Compute all loss components"""
    batch_size = t.shape[0]
    
    # Enable gradient computation for PDE terms
    t.requires_grad = True
    p.requires_grad = True
    x.requires_grad = True
    y.requires_grad = True
    
    # Forward pass
    u, Yb, Ys = model(t, p, x, y)
    
    # ===== 1. PDE Loss (No-transaction region) =====
    # Compute gradients for u
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                             create_graph=True, retain_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                             create_graph=True, retain_graph=True)[0]
    u_p = torch.autograd.grad(u, p, grad_outputs=torch.ones_like(u),
                             create_graph=True, retain_graph=True)[0]
    u_y = torch.autograd.grad(u, y, grad_outputs=torch.ones_like(u),
                             create_graph=True, retain_graph=True)[0]
    
    # Second derivatives
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                              create_graph=True, retain_graph=True)[0]
    u_xp = torch.autograd.grad(u_x, p, grad_outputs=torch.ones_like(u_x),
                              create_graph=True, retain_graph=True)[0]
    u_pp = torch.autograd.grad(u_p, p, grad_outputs=torch.ones_like(u_p),
                              create_graph=True, retain_graph=True)[0]
    
    # PDE for u (Equation 4.17 in paper)
    pde_residual = (
        u_t 
        + params.kappa * (params.theta - x) * u_x
        + params.mu * p * u_p
        + 0.5 * params.nu**2 * (u_xx + u_x**2)
        + params.rho * params.nu * params.sigma * p * (u_xp + u_p * u_x)
        + 0.5 * params.sigma**2 * p**2 * (u_pp + u_p**2)
    )
    
    # Only enforce PDE in no-transaction region (Yb ≤ y ≤ Ys)
    mask_no_tx = (y >= Yb) & (y <= Ys)
    pde_loss = torch.mean((pde_residual * mask_no_tx.float())**2)
    
    # ===== 2. Buy Region Loss =====
    mask_buy = (y < Yb)
    n_buy = mask_buy.sum().item()
    
    if n_buy > 0:
        # Get indices of buy region points
        buy_indices = mask_buy.nonzero(as_tuple=True)[0]
        
        # Extract tensors for buy region
        t_buy = t[buy_indices]
        p_buy = p[buy_indices]
        x_buy = x[buy_indices]
        y_buy = y[buy_indices]
        Yb_buy = Yb[buy_indices]
        u_buy = u[buy_indices]
        
        # Explicit solution in buy region (Equation 4.20)
        with torch.no_grad():
            u_Yb, _, _ = model(t_buy, p_buy, x_buy, Yb_buy)
        
        u_buy_explicit = u_Yb - params.gamma * torch.exp(params.r * (params.T - t_buy)) * \
                        A_minus(p_buy, x_buy) * (y_buy - Yb_buy)
        buy_loss = torch.mean((u_buy - u_buy_explicit)**2)
    else:
        buy_loss = torch.tensor(0.0, device=device, dtype=t.dtype)
    
    # ===== 3. Sell Region Loss =====
    mask_sell = (y > Ys)
    n_sell = mask_sell.sum().item()
    
    if n_sell > 0:
        # Get indices of sell region points
        sell_indices = mask_sell.nonzero(as_tuple=True)[0]
        
        # Extract tensors for sell region
        t_sell = t[sell_indices]
        p_sell = p[sell_indices]
        x_sell = x[sell_indices]
        y_sell = y[sell_indices]
        Ys_sell = Ys[sell_indices]
        u_sell = u[sell_indices]
        
        # Explicit solution in sell region (Equation 4.21)
        with torch.no_grad():
            u_Ys, _, _ = model(t_sell, p_sell, x_sell, Ys_sell)
        
        u_sell_explicit = u_Ys - params.gamma * torch.exp(params.r * (params.T - t_sell)) * \
                         A_plus(p_sell, x_sell) * (y_sell - Ys_sell)
        sell_loss = torch.mean((u_sell - u_sell_explicit)**2)
    else:
        sell_loss = torch.tensor(0.0, device=device, dtype=t.dtype)
    
    # ===== 4. Boundary Conditions (Smooth-pasting) =====
    # Sample points for boundary conditions
    bc_batch_size = max(batch_size // 10, 1)
    t_bc = torch.rand(bc_batch_size, 1, device=device, dtype=t.dtype) * params.T
    p_bc = torch.rand(bc_batch_size, 1, device=device, dtype=t.dtype) * (params.p_max - params.p_min) + params.p_min
    x_bc = torch.rand(bc_batch_size, 1, device=device, dtype=t.dtype) * (params.x_max - params.x_min) + params.x_min
    
    # Get boundaries at these points
    _, Yb_bc, Ys_bc = model(t_bc, p_bc, x_bc, torch.zeros_like(p_bc, dtype=t.dtype))
    
    # Evaluate at boundaries with gradient tracking
    # Buy boundary
    Yb_bc.requires_grad = True
    u_Yb_bc, _, _ = model(t_bc, p_bc, x_bc, Yb_bc)
    
    # Sell boundary
    Ys_bc.requires_grad = True
    u_Ys_bc, _, _ = model(t_bc, p_bc, x_bc, Ys_bc)
    
    # Compute gradients at boundaries
    grad_Yb = torch.autograd.grad(u_Yb_bc, Yb_bc, grad_outputs=torch.ones_like(u_Yb_bc),
                                  create_graph=True)[0]
    grad_Ys = torch.autograd.grad(u_Ys_bc, Ys_bc, grad_outputs=torch.ones_like(u_Ys_bc),
                                  create_graph=True)[0]
    
    # Smooth-pasting conditions (Equations 4.24-4.25)
    bc_buy_loss = torch.mean((grad_Yb + params.gamma * torch.exp(params.r * (params.T - t_bc)) * 
                             A_minus(p_bc, x_bc))**2)
    bc_sell_loss = torch.mean((grad_Ys + params.gamma * torch.exp(params.r * (params.T - t_bc)) * 
                              A_plus(p_bc, x_bc))**2)
    bc_loss = bc_buy_loss + bc_sell_loss
    
    # ===== 5. Terminal Condition Loss =====
    mask_terminal = (t > 0.99 * params.T)  # Points near terminal time
    n_terminal = mask_terminal.sum().item()
    
    if n_terminal > 0:
        # Get indices of terminal points
        terminal_indices = mask_terminal.nonzero(as_tuple=True)[0]
        
        # Extract tensors for terminal region
        t_terminal = t[terminal_indices]
        p_terminal = p[terminal_indices]
        x_terminal = x[terminal_indices]
        y_terminal = y[terminal_indices]
        u_terminal = u[terminal_indices]
        
        terminal_target = terminal_condition(p_terminal, x_terminal, y_terminal)
        terminal_loss = torch.mean((u_terminal - terminal_target)**2)
    else:
        terminal_loss = torch.tensor(0.0, device=device, dtype=t.dtype)
    
    # ===== 6. Ordering Constraint Loss =====
    # Ensure Y_b <= Y_s
    ordering_loss = torch.mean(torch.relu(Yb - Ys)**2)
    
    # ===== Total Loss =====
    weights = {
        'pde': 1.0,
        'buy': 10.0,
        'sell': 10.0,
        'bc': 5.0,
        'terminal': 5.0,
        'ordering': 1.0
    }
    
    total_loss = (
        weights['pde'] * pde_loss +
        weights['buy'] * buy_loss +
        weights['sell'] * sell_loss +
        weights['bc'] * bc_loss +
        weights['terminal'] * terminal_loss +
        weights['ordering'] * ordering_loss
    )
    
    return {
        'total': total_loss,
        'pde': pde_loss,
        'buy': buy_loss,
        'sell': sell_loss,
        'bc': bc_loss,
        'terminal': terminal_loss,
        'ordering': ordering_loss
    }

# ====================
# Training Function
# ====================
def train_pinn(model, epochs=35000, batch_size=12000, lr=1e-3, device='cpu'):
    """Train the PINN"""
    # Convert model to double precision
    model = model.double().to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    # Cosine annealing scheduler with warm restarts
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5000, T_mult=2, eta_min=1e-5
    )
    
    losses_history = {key: [] for key in ['total', 'pde', 'buy', 'sell', 'bc', 'terminal', 'ordering']}
    
    print(f"Training PINN on {device}...")
    start_time = time.time()
    
    for epoch in tqdm(range(epochs)):
        # Sample new collocation points each epoch
        t, p, x, y = sample_domain(batch_size, device, dtype=torch.float64)
        
        # Zero gradients
        optimizer.zero_grad()
        
        # Compute losses
        losses = compute_losses(model, t, p, x, y, device)
        
        # Backward pass with gradient clipping
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # Update weights
        optimizer.step()
        scheduler.step()
        
        # Record losses
        for key in losses_history:
            if key in losses:
                losses_history[key].append(losses[key].item())
        
        # Print progress every 1000 epochs
        if (epoch + 1) % 1000 == 0:
            print(f"Epoch {epoch+1}/{epochs}, Total Loss: {losses['total'].item():.6f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return losses_history, model

# ====================
# Visualization Functions
# ====================
def plot_buy_sell_boundaries(model, device='cpu'):
    """Plot buy and sell boundaries at different times"""
    model.eval()
    
    # Create grid for plotting
    p_values = np.linspace(params.p_min, params.p_max, 30)
    x_values = np.linspace(params.x_min, params.x_max, 30)
    p_grid, x_grid = np.meshgrid(p_values, x_values)
    
    # Times to plot
    times = [0.05, 0.35, 0.65, 0.95]
    
    fig = plt.figure(figsize=(15, 10))
    
    for i, t_val in enumerate(times):
        ax = fig.add_subplot(2, 2, i+1, projection='3d')
        
        # Convert to tensors
        t_tensor = torch.full((p_grid.size, 1), t_val, dtype=torch.float64, device=device)
        p_tensor = torch.tensor(p_grid.reshape(-1, 1), dtype=torch.float64, device=device)
        x_tensor = torch.tensor(x_grid.reshape(-1, 1), dtype=torch.float64, device=device)
        y_dummy = torch.zeros_like(p_tensor, dtype=torch.float64, device=device)
        
        # Get boundaries
        with torch.no_grad():
            _, Yb, Ys = model(t_tensor, p_tensor, x_tensor, y_dummy)
        
        # Reshape for plotting
        Yb_plot = Yb.cpu().numpy().reshape(p_grid.shape)
        Ys_plot = Ys.cpu().numpy().reshape(p_grid.shape)
        
        # Plot buy boundary (blue)
        ax.plot_surface(p_grid, x_grid, Yb_plot, alpha=0.6, color='blue')
        
        # Plot sell boundary (red)
        ax.plot_surface(p_grid, x_grid, Ys_plot, alpha=0.6, color='red')
        
        ax.set_xlabel('Price (p)')
        ax.set_ylabel('Spread (x)')
        ax.set_zlabel('Inventory (y)')
        ax.set_title(f'Time t = {t_val}')
        ax.view_init(elev=30, azim=45)
    
    plt.suptitle('Buy and Sell Boundaries at Different Times', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_loss_history(losses_history):
    """Plot training loss history"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    loss_types = ['total', 'pde', 'buy', 'sell', 'bc', 'terminal']
    titles = ['Total Loss', 'PDE Loss', 'Buy Region Loss', 
              'Sell Region Loss', 'Boundary Condition Loss', 'Terminal Condition Loss']
    
    for idx, (loss_type, title) in enumerate(zip(loss_types, titles)):
        ax = axes[idx]
        if loss_type in losses_history and len(losses_history[loss_type]) > 0:
            # Plot on log scale for better visibility
            ax.semilogy(losses_history[loss_type])
            ax.set_title(title)
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.grid(True, alpha=0.3)
    
    plt.suptitle('Training Loss History', fontsize=16)
    plt.tight_layout()
    plt.show()

def plot_simple_strategy_slice(model, device='cpu'):
    """Plot a 2D slice of the optimal strategy for fixed t and p"""
    model.eval()
    
    # Fixed parameters for slice
    t_fixed = 0.5
    p_fixed = 1.0
    
    # Create grid
    x_values = np.linspace(params.x_min, params.x_max, 30)
    y_values = np.linspace(params.y_min, params.y_max, 30)
    
    # Compute boundaries along x-axis
    Yb_values = np.zeros_like(x_values)
    Ys_values = np.zeros_like(x_values)
    
    for i, x_val in enumerate(x_values):
        t_tensor = torch.tensor([[t_fixed]], dtype=torch.float64, device=device)
        p_tensor = torch.tensor([[p_fixed]], dtype=torch.float64, device=device)
        x_tensor = torch.tensor([[x_val]], dtype=torch.float64, device=device)
        y_dummy = torch.zeros_like(p_tensor, dtype=torch.float64, device=device)
        
        with torch.no_grad():
            _, Yb, Ys = model(t_tensor, p_tensor, x_tensor, y_dummy)
            Yb_values[i] = Yb.item()
            Ys_values[i] = Ys.item()
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot boundaries
    ax.plot(x_values, Yb_values, 'b-', linewidth=2, label='Buy boundary (Yb)')
    ax.plot(x_values, Ys_values, 'r-', linewidth=2, label='Sell boundary (Ys)')
    
    # Fill regions
    ax.fill_between(x_values, params.y_min, Yb_values, alpha=0.2, color='blue', label='Buy region')
    ax.fill_between(x_values, Yb_values, Ys_values, alpha=0.2, color='green', label='No-transaction region')
    ax.fill_between(x_values, Ys_values, params.y_max, alpha=0.2, color='red', label='Sell region')
    
    ax.set_xlabel('Spread (x)')
    ax.set_ylabel('Inventory (y)')
    ax.set_title(f'Trading Regions at t={t_fixed}, p={p_fixed}')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# ====================
# Main Execution
# ====================
if __name__ == "__main__":
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Initialize PINN model
    print("Initializing PINN model...")
    model = PINN(n_layers=6, n_neurons=64)
    
    # Train the model (with fewer epochs for testing)
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50)
    
    batch_size = 5000  # Reduced for testing
    epochs = 100  # Reduced for testing
    
    losses_history, trained_model = train_pinn(
        model, 
        epochs=epochs, 
        batch_size=batch_size,
        lr=1e-3,
        device=device
    )
    
    # Visualize results
    print("\n" + "="*50)
    print("Generating plots...")
    print("="*50)
    
    # Plot training loss history
    plot_loss_history(losses_history)
    
    # Plot buy and sell boundaries
    plot_buy_sell_boundaries(trained_model, device=device)
    
    # Plot simple strategy slice
    plot_simple_strategy_slice(trained_model, device=device)
    
    # Print final losses
    print("\nFinal Losses:")
    for key, values in losses_history.items():
        if values:
            print(f"  {key}: {values[-1]:.6f}")
    
    # Save the trained model
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'params': params,
        'losses': losses_history
    }, 'pairs_trading_pinn.pth')
    print("\nModel saved to 'pairs_trading_pinn.pth'")