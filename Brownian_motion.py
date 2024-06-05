import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#simulation of stockprices with antithetic paths
def sim_stock_prices(simnum, sigma, mu, N, T, S0):
    dt = 1/N
    Stock_price_paths=np.zeros((simnum, (T*N)+1))
    # Generate stock prices
    Stock_price_paths[:, 0] = S0
    for i in range(0,simnum,2):
        for j in range(1,(T*N)+1):
            z=np.random.normal(loc=0, scale=1)
            Stock_price_paths[i,j] = Stock_price_paths[i,j-1] * np.exp((mu - sigma ** 2 / 2) * dt + sigma *z*np.sqrt(dt))
            Stock_price_paths[i+1, j] = Stock_price_paths[i+1, j - 1] * np.exp((mu - sigma ** 2 / 2) * dt - sigma *z*np.sqrt(dt))
    return Stock_price_paths

#simulation without Antithetic paths
def sim_stock_prices2(simnum, sigma, mu, N, T, S0):
    dt = 1/N
    Stock_price_paths=np.zeros((simnum, (T*N)+1))
    # Generate stock prices
    Stock_price_paths[:, 0] = S0
    for i in range(0,simnum,1):
        for j in range(1,(T*N)+1):
            z=np.random.normal(loc=0, scale=1)
            Stock_price_paths[i,j] = Stock_price_paths[i,j-1] * np.exp((mu - sigma ** 2 / 2) * dt + sigma *z*np.sqrt(dt))
    return Stock_price_paths

#Simulate stockprices
#simnum: number of simulates paths, sigma: Volatility, mu: short interest rate, N: Exercise points till exercise
#S0: initial stock price, T: Maturity time in years, N0: inital exercise points.
def sim_stock_pricesflex(simnum, sigma, mu, N, S0,T, N0):
    dt=1/N0
    Stock_price_paths=np.zeros((simnum, (T*N)+1))
    # Generate stock prices
    Stock_price_paths[:, 0] = S0
    for i in range(0,simnum,2):
        for j in range(1,(T*N)+1):
            z=np.random.normal(loc=0, scale=1)
            Stock_price_paths[i,j] = Stock_price_paths[i,j-1] * np.exp((mu - sigma ** 2 / 2) * dt + sigma *z*np.sqrt(dt))
            Stock_price_paths[i+1, j] = Stock_price_paths[i+1, j - 1] * np.exp((mu - sigma ** 2 / 2) * dt - sigma *z*np.sqrt(dt))
    return Stock_price_paths

def GBM_plot2(simnum, sigma, mu, N, T, S0):
    sim_stock_prices = sim_stock_prices2(1, sigma, mu, N, T, S0)
    dt = 1 / N
    X =  np.arange(0, T+dt, dt)
    sns.set(style="darkgrid", palette="Greens_d")
    plt.plot(X,sim_stock_prices.T)
    plt.xlabel('Time')
    plt.ylabel('Stock price')
    plt.title('Geometric Brownian Motion')
    plt.xticks(np.arange(0, T + dt, 0.1))
    #plt.legend()
    plt.show()

def GBM_plot(simnum, sigma, mu, N, T, S0):
    stock_prices = sim_stock_prices(simnum, sigma, mu, N, T, S0)
    dt = 1 / N
    X =  np.arange(0, T+dt, dt)
    plt.plot(X,stock_prices.T)
    plt.xlabel('Time')
    plt.ylabel('Stock price')
    plt.title('Geometric Brownian Motion')
    plt.grid()
    plt.show()

#GBM_plot2(5, 0.2, 0.06, 200, 1, 36)
#GBM_plot(6, 0.2, 0.06, 50, 1, 36)

simnum = 5
sigma = 0.2
mu = 0.06 #interest rate if under Q.
N = 50
T = 1
S0 = 36
K = 44
alpha=0.5
