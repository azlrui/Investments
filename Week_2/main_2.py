import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import inv
from scipy.stats import norm


# Example parameters
return_aapl = 0.29
return_pg = 0.09
std_aapl = 0.38
std_pg = 0.2
corr = 0.1  # Correlation between AAPL and PG
rf = 0.015

# Define portfolio return and standard deviation
def return_p(w):
    return w * return_aapl + (1 - w) * return_pg

def std_p(w):
    return np.sqrt(w**2 * std_aapl**2 + (1 - w)**2 * std_pg**2 + 2 * w * (1 - w) * corr * std_pg * std_aapl)

# Generate portfolio combinations for weights from 0 to 1
weights = np.linspace(0, 1, 100)
vector_returns = [return_p(w) for w in weights]
vector_std = [std_p(w) for w in weights]

# Plot efficient frontier
plt.figure(figsize=(8,5))
plt.plot(vector_std, vector_returns, label="Mean-Std Dev Frontier", color="b")
plt.xlabel("Portfolio Standard Deviation")
plt.ylabel("Portfolio Expected Return")
plt.title("Mean-Standard Deviation Frontier (Two Assets)")
plt.legend()
plt.grid()
plt.show()

### Question (c) - Optimal Portfolio
s_opt = np.array([0.1915, 0.8085])
return_P_opt = s_opt @ np.array([return_aapl, return_pg])

cov_matrix = np.array([[0.1444, 0.0076], [0.0076, 0.04]])
var_dev_P = s_opt @ cov_matrix @ s_opt.T

sharpe_ratio = (return_P_opt - rf) / np.sqrt(var_dev_P)

print("Mean:", return_P_opt)
print("Std.dev:", np.sqrt(var_dev_P))
print("Sharpe Ratio:", sharpe_ratio)

### Question (d) - Tangency Portfolio
cov_var_matrix = np.array([[std_aapl**2, std_aapl*std_pg*corr], [std_aapl*std_pg*corr, std_pg**2]])
inv_cov_var = inv(cov_var_matrix)
ret = np.array([return_aapl, return_pg])
one = np.ones(2)

num = inv_cov_var @ (ret - rf * one)
w_tan = num / (one @ num)

tan_w_aapl = w_tan[0]
tan_w_pg = w_tan[1]
tan_ret = w_tan @ ret
tan_std = np.sqrt(w_tan.T @ cov_var_matrix @ w_tan)
SR_tan = (w_tan.T @ ret - rf) / np.sqrt(w_tan.T @ cov_var_matrix @ w_tan)

print("\n")
print("\n")
print("Tangent portfolio:")
print("Weight in AAPL:", tan_w_aapl)
print("Weight in PG:", tan_w_pg)
print("Weight in portfolio:", 1)
print("Return:", tan_ret)
print("Std.dev:", tan_std)
print("Sharpe Ratio:", SR_tan)

# Capital Market Line (CML)
x = np.linspace(start=0, stop=0.5, num=10000)
y = rf + SR_tan * x

plt.scatter(np.sqrt(w_tan.T @ cov_var_matrix @ w_tan), w_tan @ ret, color="black", marker="o", label="Tangency Portfolio")

plt.plot(vector_std, vector_returns, label="Mean-Std Dev Frontier", color="b")
plt.plot(x, y, label="Capital Market Line (CML)", color="r", linestyle="--")

plt.grid()
plt.show()

#Question (f)
target_ret = 0.2
on_tan_weight = target_ret / tan_std

new_tan_w_aapl = tan_w_aapl * on_tan_weight
new_tan_w_pg = tan_w_pg * on_tan_weight
new_tan_rf = (1 - new_tan_w_pg - new_tan_w_aapl)

new_tan_w = np.array([new_tan_w_aapl, new_tan_w_pg])

aversion_vector = inv_cov_var @ (ret - rf * np.ones(2))


tan_std_new = np.sqrt(new_tan_w.T @ cov_var_matrix @ new_tan_w)
tan_mean_new = rf * new_tan_rf + new_tan_w @ ret
new_tan_rs = (tan_mean_new - rf) / tan_std_new

print("\n")
print("\n")
print("New portfolio:")
print("<AAPL>:", new_tan_w_aapl)
print("<PG>:", new_tan_w_pg)
print("<risk-free asset>:", new_tan_rf)
print("<Return>:", tan_mean_new)
print("<Std>:", tan_std_new)
print("<Sharp Ratio>:", new_tan_rs)
print("<Aversion> :", aversion_vector[0]/new_tan_w_aapl)

#Question (g)
inv_one_percent = norm.ppf(0.01)

share = (-0.10 - rf) / (tan_ret + tan_std*inv_one_percent - rf)

aapl_w = tan_w_aapl * share
pg_w = tan_w_pg * share
rf_w = 1 - aapl_w - pg_w

expected_return = rf_w * rf + aapl_w * return_aapl + rf_w * return_pg
volatility = np.sqrt(aapl_w**2 * std_aapl**2 + pg_w**2 * std_pg**2 + 2 * corr * std_pg * std_aapl * aapl_w * pg_w)

sr = (expected_return - rf)/volatility

a = inv_cov_var @ (ret - np.ones(2) * rf).T
a = a[0]/aapl_w

print("\n")
print("\n")
print("Portfolio:")
print("<share in tan>:", share)
print("<AAPL>:", aapl_w)
print("<PG>:", pg_w)
print("<risk-free asset>:", rf_w)
print("<Return>:", expected_return)
print("<Std>:", volatility)
print("<Sharp Ratio>:", sr)
print("<Aversion> :", a)