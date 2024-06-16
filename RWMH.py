import numpy as np
from scipy.special import gammaln
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

def extract_nx(data):
    return np.array([t[1] for t in data]), np.array([t[0] for t in data])

# Log Posterior
def log_posterior(theta, data):
    alpha, beta = theta
    n, x = extract_nx(data)
    log_prior = -2.5 * np.log(alpha + beta)
    log_likelihood = np.sum(gammaln(alpha + beta) - gammaln(alpha) - gammaln(beta) + gammaln(alpha + x) + gammaln(beta + n - x) - gammaln(alpha + beta + n))
    return log_prior + log_likelihood

# RandomWalk Metropolis-Hastings Sampling
def metropolis_hastings(log_post, initial, iterations, data, variance=0.1):
    samples = np.zeros((iterations, 2))
    current = np.array(initial)
    current_log_post = log_post(current, data)
    
    for i in range(iterations):
        proposal = current + np.random.normal(0, variance, 2)
        proposal_log_post = log_post(proposal, data)
        acceptance = np.exp(proposal_log_post - current_log_post)
        
        if acceptance >= np.random.rand():
            current = proposal
            current_log_post = proposal_log_post
        
        samples[i, :] = current
    
    return samples, acceptance

# 샘플링 시작
initial_guess = [1.4, 8.6]
iterations = 10000  # 반복 횟수
variance = 0.01


start_time = time.time()
samples, acceptance = metropolis_hastings(log_posterior, initial_guess, iterations, data, variance)
end_time = time.time()

print(f"Runtime : {end_time-start_time}s")
print(f"Acceptance Rate : {acceptance}")
mean = np.mean(samples, axis=0)
print(f"Mean of (alpha, beta) : ({mean[0]}, {mean[1]})")
cov = np.cov(samples, rowvar=False)
print('Covaraince matrix of MH sampling data')
print(cov)

# Plotting the results
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.5)
plt.xlabel('alpha')
plt.ylabel('beta')
plt.title('Random Walk Metropolis Hastings Sampling of alpha and beta')
plt.grid(True)
plt.show()

burn_in = 100
after_burn_in = samples[burn_in:,:]
ess = []
for i in range(after_burn_in.shape[1]):
    # 자기상관 계수 계산
    autocorr = acf(after_burn_in[:, i], fft=True, nlags=40)
    # ESS 계산
    ess.append(after_burn_in.shape[0] / (1 + 2 * np.sum(autocorr[1:])))
print(ess)