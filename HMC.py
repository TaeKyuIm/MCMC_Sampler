import numpy as np
from scipy.special import gammaln, psi
import time
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import acf

np.random.seed(42)

# Data
data = [
    (0, 20), (0, 20), (0, 20), (0, 20), (0, 20), (0, 20), (0, 20), (0, 19), (0, 19), (0, 19),
    (0, 19), (0, 18), (0, 18), (0, 17), (1, 20), (1, 20), (1, 20), (1, 20), (1, 19), (1, 19),
    (1, 18), (1, 18), (2, 25), (2, 24), (2, 23), (2, 20), (2, 20), (2, 20), (2, 20), (2, 20),
    (2, 20), (1, 10), (5, 49), (2, 19), (5, 46), (3, 27), (2, 17), (7, 49), (7, 47), (3, 20),
    (3, 20), (2, 13), (9, 48), (10, 50), (4, 20), (4, 20), (4, 20), (4, 20), (4, 20), (4, 20),
    (4, 20), (10, 48), (4, 19), (4, 19), (4, 19), (5, 22), (11, 46), (12, 49), (5, 20), (5, 20),
    (6, 23), (5, 19), (6, 22), (6, 20), (6, 20), (6, 20), (16, 52), (15, 47), (15, 46), (9, 24),
    (4, 14)
]

# Extract trials and successes
def extract_nx(data):
    n = np.array([t[1] for t in data])
    x = np.array([t[0] for t in data])
    return n, x

# Log-posterior function for Bayesian inference
def log_posterior(theta, data):
    alpha, beta = theta
    n, x = extract_nx(data)
    log_prior = -2.5 * np.log(alpha + beta)
    log_likelihood = np.sum(gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta) +
                            gammaln(alpha + x) + gammaln(beta + n - x) - gammaln(alpha + beta + n))
    return log_prior + log_likelihood

# Gradient of the log-posterior function
def log_posterior_grad(theta, data):
    alpha, beta = theta
    n, x = extract_nx(data)
    grad_alpha = -2.5 / (alpha + beta) + np.sum(psi(alpha + beta) - psi(alpha) + psi(alpha + x) - psi(alpha + beta + n))
    grad_beta = -2.5 / (alpha + beta) + np.sum(psi(alpha + beta) - psi(beta) + psi(beta + n - x) - psi(alpha + beta + n))
    return np.array([grad_alpha, grad_beta])

# Hamiltonian Monte Carlo sampling function
def hamiltonian_monte_carlo(log_post, log_post_grad, initial, iterations, data, step_size=0.01, L=10, M=1):
    samples = np.zeros((iterations, 2))
    current = np.array(initial)
    current_log_post = log_post(current, data)
    acceptances = 0

    for i in range(iterations):
        p_current = np.random.normal(0, M, 2)
        theta = current.copy()
        p_new = p_current.copy()

        for _ in range(L):
            grad = log_post_grad(theta, data)
            p_new -= 0.5 * step_size * grad
            theta += step_size * p_new
            grad = log_post_grad(theta, data)
            p_new -= 0.5 * step_size * grad

        proposed_log_post = log_post(theta, data)
        proposed_p = -np.sum(p_new**2) / 2
        current_p = -np.sum(p_current**2) / 2

        if np.random.rand() < np.exp(proposed_log_post + proposed_p - current_log_post - current_p):
            current = theta
            current_log_post = proposed_log_post
            acceptances += 1

        samples[i, :] = current

    return samples, acceptances/iterations

# Set up parameters for the sampling
n, x = extract_nx(data)
initial_guess = [1.4, 8.6]
iterations = 10000
step_size = 0.005
L = 30
M = 1

# Start timing, run the sampling, and stop timing
start_time = time.time()
samples, acceptance_rate = hamiltonian_monte_carlo(log_posterior, log_posterior_grad, initial_guess, iterations, data, step_size, L, M)
end_time = time.time()

print(f"Runtime : {end_time - start_time}s")
print(f"Acceptance Rate : {acceptance_rate}")

# Compute and print statistics
mean_alpha_beta = np.mean(samples, axis=0)
print(f"Mean of (alpha, beta) : ({mean_alpha_beta[0]}, {mean_alpha_beta[1]})")
cov = np.cov(samples, rowvar=False)
print(cov)

# Plotting the results
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.title('HMC Sampling of alpha and beta')
plt.grid(True)
plt.show()

burn_in = 100
after_burn_in = samples[burn_in:,:]
print(after_burn_in.shape[1])
ess = []
for i in range(after_burn_in.shape[1]):
    # 자기상관 계수 계산
    autocorr = acf(after_burn_in[:, i], fft=True, nlags=40)
    # ESS 계산
    ess.append(after_burn_in.shape[0] / (1 + 2 * np.sum(autocorr[1:])))
print(ess)