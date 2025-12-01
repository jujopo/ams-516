import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

class PairsTradingPINN(nn.Module):
    def __init__(self, layers, params):
        super(PairsTradingPINN, self).__init__()
        
        # Network architecture
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            if i < len(layers)-2:
                self.layers.append(nn.Tanh())
        
        # Model parameters
        self.mu = params['mu']
        self.sigma = params['sigma']
        self.kappa = params['kappa']
        self.theta = params['theta']
        self.nu = params['nu']
        self.rho = params['rho']
        self.r = params['r']
        self.gamma = params['gamma']
        
        # Transaction cost parameters
        self.a_p = 1 + params['zeta_p']
        self.b_p = 1 - params['eta_p']
        self.a_q = 1 + params['zeta_q']
        self.b_q = 1 - params['eta_q']
        
        self.T = params['T']
        
    def forward(self, x):
        # x: [t, p, x, y]
        for layer in self.layers:
            x = layer(x)
        return x
    
    def compute_derivatives(self, t, p, x, y):
        """
        Compute all required derivatives for the HJB equation
        """
        # Enable gradient computation
        t.requires_grad_(True)
        p.requires_grad_(True)
        x.requires_grad_(True)
        y.requires_grad_(True)
        
        # Forward pass
        inputs = torch.cat([t, p, x, y], dim=1)
        H = self.forward(inputs)
        
        # First derivatives
        dH_dt = torch.autograd.grad(H, t, grad_outputs=torch.ones_like(H), 
                                   create_graph=True)[0]
        dH_dp = torch.autograd.grad(H, p, grad_outputs=torch.ones_like(H), 
                                   create_graph=True)[0]
        dH_dx = torch.autograd.grad(H, x, grad_outputs=torch.ones_like(H), 
                                   create_graph=True)[0]
        dH_dy = torch.autograd.grad(H, y, grad_outputs=torch.ones_like(H), 
                                   create_graph=True)[0]
        
        # Second derivatives
        d2H_dx2 = torch.autograd.grad(dH_dx, x, grad_outputs=torch.ones_like(dH_dx), 
                                     create_graph=True)[0]
        
        d2H_dpdx = torch.autograd.grad(dH_dp, x, grad_outputs=torch.ones_like(dH_dp), 
                                      create_graph=True)[0]
        
        d2H_dp2 = torch.autograd.grad(dH_dp, p, grad_outputs=torch.ones_like(dH_dp), 
                                     create_graph=True)[0]
        
        return H, dH_dt, dH_dp, dH_dx, dH_dy, d2H_dx2, d2H_dpdx, d2H_dp2
    
    def A_plus(self, p, x):
        """A_+(p,x) = (b_p - a_q * exp(x)) * p"""
        return (self.b_p - self.a_q * torch.exp(x)) * p
    
    def A_minus(self, p, x):
        """A_-(p,x) = (a_p - b_q * exp(x)) * p"""
        return (self.a_p - self.b_q * torch.exp(x)) * p
    
    def L2_o(self, H, dH_dt, dH_dp, dH_dx, d2H_dx2, d2H_dpdx, d2H_dp2, t, p, x):
        """Operator L_{2,o} from equation (25)"""
        term1 = dH_dt
        term2 = self.kappa * (self.theta - x) * dH_dx
        term3 = self.mu * p * dH_dp
        term4 = 0.5 * self.nu**2 * d2H_dx2
        term5 = self.rho * self.nu * self.sigma * p * d2H_dpdx
        term6 = 0.5 * self.sigma**2 * p**2 * d2H_dp2
        
        return term1 + term2 + term3 + term4 + term5 + term6
    
    def L2_b(self, H, dH_dy, t, p, x):
        """Operator L_{2,b} from equation (25)"""
        return dH_dy + self.gamma * torch.exp(self.r * (self.T - t)) * self.A_minus(p, x) * H
    
    def L2_s(self, H, dH_dy, t, p, x):
        """Operator L_{2,s} from equation (25)"""
        return dH_dy + self.gamma * torch.exp(self.r * (self.T - t)) * self.A_plus(p, x) * H
    
    def HJB_loss(self, t, p, x, y):
        """
        Compute the HJB PDE loss from equation (25):
        min{L2_b H, -L2_s H, L2_o H} = 0
        """
        H, dH_dt, dH_dp, dH_dx, dH_dy, d2H_dx2, d2H_dpdx, d2H_dp2 = self.compute_derivatives(t, p, x, y)
        
        L2o_H = self.L2_o(H, dH_dt, dH_dp, dH_dx, d2H_dx2, d2H_dpdx, d2H_dp2, t, p, x)
        L2b_H = self.L2_b(H, dH_dy, t, p, x)
        L2s_H = self.L2_s(H, dH_dy, t, p, x)
        
        # The HJB equation: min{L2_b H, -L2_s H, L2_o H} = 0
        # We implement this as three separate constraints
        loss_pde = (torch.relu(L2b_H)**2 +  # L2_b H >= 0
                   torch.relu(L2s_H)**2 +    # L2_s H <= 0 (so -L2_s H >= 0)
                   torch.relu(L2o_H)**2)     # L2_o H >= 0
        
        return loss_pde.mean()
    
    def terminal_condition_loss(self, p, x, y):
        """
        Terminal condition at t=T:
        H(T, p, x, y) = exp(-gamma * J(p, x, y))
        """
        t_T = torch.ones_like(p) * self.T  # Terminal time
        
        inputs = torch.cat([t_T, p, x, y], dim=1)
        H_pred = self.forward(inputs)
        
        # Compute J(p, x, y) from equation (10)
        A_plus = self.A_plus(p, x)
        A_minus = self.A_minus(p, x)
        
        # Indicator functions for y >= 0 and y < 0
        y_positive = (y >= 0).float()
        y_negative = (y < 0).float()
        
        J = A_plus * y * y_positive + A_minus * y * y_negative
        
        H_true = torch.exp(-self.gamma * J)
        
        return torch.mean((H_pred - H_true)**2)
    
    def total_loss(self, interior_points, boundary_points):
        """
        Total loss combining PDE and boundary conditions
        """
        # Unpack interior points
        t_int, p_int, x_int, y_int = interior_points
        
        # Unpack boundary points (terminal condition)
        p_bc, x_bc, y_bc = boundary_points
        
        # PDE loss
        loss_pde = self.HJB_loss(t_int, p_int, x_int, y_int)
        
        # Terminal condition loss
        loss_bc = self.terminal_condition_loss(p_bc, x_bc, y_bc)
        
        return loss_pde + loss_bc

def train_pinn(model, num_epochs, lr, domain_bounds, num_points=1000):
    """
    Train the PINN model
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    losses = []
    
    for epoch in range(num_epochs):
        # Generate training points
        interior_points = generate_interior_points(domain_bounds, num_points)
        boundary_points = generate_boundary_points(domain_bounds, num_points // 10)
        
        optimizer.zero_grad()
        
        loss = model.total_loss(interior_points, boundary_points)
        
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    return losses

def generate_interior_points(domain_bounds, num_points):
    """
    Generate random points in the interior domain
    """
    t_min, t_max, p_min, p_max, x_min, x_max, y_min, y_max = domain_bounds
    
    t = torch.rand(num_points, 1) * (t_max - t_min) + t_min
    p = torch.rand(num_points, 1) * (p_max - p_min) + p_min
    x = torch.rand(num_points, 1) * (x_max - x_min) + x_min
    y = torch.rand(num_points, 1) * (y_max - y_min) + y_min
    
    return t, p, x, y

def generate_boundary_points(domain_bounds, num_points):
    """
    Generate random points on the terminal boundary (t=T)
    """
    _, _, p_min, p_max, x_min, x_max, y_min, y_max = domain_bounds
    
    p = torch.rand(num_points, 1) * (p_max - p_min) + p_min
    x = torch.rand(num_points, 1) * (x_max - x_min) + x_min
    y = torch.rand(num_points, 1) * (y_max - y_min) + y_min
    
    return p, x, y

# Example usage
if __name__ == "__main__":
    # Model parameters from the paper (Scenario S1)
    params = {
        'mu': 0.2,
        'sigma': 0.4,
        'kappa': 1.0,
        'theta': 0.1,
        'nu': 0.15,
        'rho': 0.5,
        'r': 0.01,
        'gamma': 5.0,
        'zeta_p': 0.0005, 'eta_p': 0.0005,
        'zeta_q': 0.0005, 'eta_q': 0.0005,
        'T': 1.0
    }
    
    # Domain bounds [t_min, t_max, p_min, p_max, x_min, x_max, y_min, y_max]
    domain_bounds = [0.0, 1.0, 0.5, 2.0, -0.5, 0.5, -10.0, 10.0]
    
    # Neural network architecture [input_dim, hidden1, hidden2, ..., output_dim]
    layers = [4, 50, 50, 50, 1]  # Input: (t, p, x, y), Output: H
    
    # Initialize model
    model = PairsTradingPINN(layers, params)
    
    # Train the model
    losses = train_pinn(model, num_epochs=1000, lr=1e-3, 
                       domain_bounds=domain_bounds, num_points=1000)
    
    # Plot training loss
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('PINN Training Loss')
    plt.grid(True)
    plt.show()