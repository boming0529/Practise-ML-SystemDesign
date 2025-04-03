import numpy as np
import matplotlib.pyplot as plt

def loss_fcn(w):
    return w**4 - 2.5 * w**3 + 1.2 * w**2 + 1

def grad_fcn(w):
    return 4 * w**3 - 7.5 * w**2 + 2.4 * w


learning_rate = 0.05 
epochs = 50
w = np.random.uniform(-1, 2) # init value in [-1, 1]

w_history = [w]
loss_history = [loss_fcn(w)]

# Gradient Descent
for epoch in range(epochs):
    grad = grad_fcn(w)       
    # w ← w − η⋅dL/dw
    w = w - learning_rate * grad 
    w = np.clip(w, -1, 2)        
    loss = loss_fcn(w)
    w_history.append(w)
    loss_history.append(loss)
    print(f"Epoch {epoch+1}: w = {w:.4f}, Loss = {loss:.4f}")

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