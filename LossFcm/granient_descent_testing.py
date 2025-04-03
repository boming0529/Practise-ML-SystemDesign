import numpy as np
import matplotlib.pyplot as plt

def loss_fcn(w):
    return w**2

def grad_fcn(w):
    return 2 * w

learning_rate = 0.1
epochs = 20
w = np.random.uniform(-1, 1) # init value in [-1, 1]

w_history = [w]
loss_history = [loss_fcn(w)]

# Gradient Descent
for epoch in range(epochs):
    grad = grad_fcn(w)            
    w = w - learning_rate * grad 
    w = np.clip(w, -1, 1)        
    loss = loss_fcn(w)
    w_history.append(w)
    loss_history.append(loss)
    print(f"Epoch {epoch+1}: w = {w:.4f}, Loss = {loss:.4f}")

w_range = np.linspace(-1, 1, 100)
loss_range = loss_fcn(w_range)
 # show all w loss function
plt.plot(w_range, loss_range, label="L(w) = w^2")  
 # show Gradient Descent loss function                  
plt.plot(w_history, loss_history, 'ro-', label="Gradient Descent Path") 
plt.xlabel("w")
plt.ylabel("Loss")
plt.legend()
plt.title("Gradient Descent on L(w) = w^2")
plt.show()