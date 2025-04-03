import torch
import numpy as np
import matplotlib.pyplot as plt

def loss_fcn(w):
    return w**4 - 2.5 * w**3 + 1.2 * w**2 + 1

def grad_fcn(w):
    return 4 * w**3 - 7.5 * w**2 + 2.4 * w


learning_rate = 0.05 
epochs = 50
w = torch.tensor([np.random.uniform(-1, 2)], requires_grad=True)

w_history = [w.item()]
loss_history = [loss_fcn(w).item()]

# Gradient Descent
for epoch in range(epochs):
    loss = loss_fcn(w)
    loss.backward()
    with torch.no_grad():
        w -= learning_rate * w.grad
        w.clamp_(-1, 2) 
        w.grad.zero_()
    w_history.append(w.item())
    loss_history.append(loss_fcn(w).item())
    print(f"Epoch {epoch+1}: w = {w.item():.4f}, Loss = {loss.item():.4f}")


w_range = np.linspace(-1, 2, 100)
loss_range = loss_fcn(w_range)
 # show all w loss function
plt.plot(w_range, loss_range, label="L(w) = w^4 - 2.5w^3 + 1.2w^2 + 1")
 # show Gradient Descent loss function                  
plt.plot(w_history, loss_history, 'ro-', label="Gradient Descent Path") 
plt.xlabel("w")
plt.ylabel("Loss")
plt.legend()
plt.title("Gradient Descent on Complex Polynomial")
plt.show()