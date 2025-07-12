import numpy as np
import matplotlib.pyplot as plt


# Gaussian PDF function
def gaussian_pdf(x, mu, sigma):
    coef = 1 / (sigma * np.sqrt(2 * np.pi))
    exponent = np.exp(- (x - mu)**2 / (2 * sigma**2))
    return coef * exponent


# Parameters
mu = 1.52      # mean
sigma = np.sqrt(0.1416)       # standard deviation

# Generate x values
x = np.linspace(-5, 5, 500)
y = gaussian_pdf(x, mu, sigma)

# Plot
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=f'Gaussian PDF\nμ={mu}, σ={sigma}')
plt.title('Gaussian Probability Density Function')
plt.xlabel('x')
plt.ylabel('Probability Density')
plt.grid(True)
plt.legend()
plt.show()
