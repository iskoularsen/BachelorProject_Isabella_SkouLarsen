import statsmodels.api as sm
from regression import *
import seaborn as sns
def ISD(simnum, S0, alpha):
    U = np.zeros(simnum)
    U_2 = np.random.uniform(0, 1, (simnum // 2 ,))
    for i in range(simnum // 2):
        U[2 * i] = U_2[i]
        U[2 * i + 1] = U_2[i]
    K_isd = 2 * np.sin(np.arcsin(2 * U - 1)/3)
    X = S0 + alpha * K_isd
    return X

#plot over aktieprisstigerne
def ISD_pricepaths_plot(simnum, sigma, mu, N, S0,T, alpha):
    Stock_price_paths =sim_stock_prices_ISD(simnum, sigma, mu, N, S0,T, alpha)
    dt = 1 / N
    X =  np.arange(0, T+dt, dt)
    sns.set(style="darkgrid",palette="Set2")
    plt.plot(X,Stock_price_paths.T)
    plt.xlabel('Time')
    plt.ylabel('Stock price')
    plt.title('ISD Stock Price Paths')
    plt.xticks(np.arange(0, T + dt, 0.1))
    #plt.legend()
    plt.show()

#plot over fordelingen af startværdier
def ISD_distribution_plot(simnum, S0, alpha):
    X = ISD(simnum, S0, alpha)
    sns.set(style="darkgrid",palette="Set2")
    plt.hist(X, bins=35)
    plt.xlabel('Stock price')
    plt.ylabel('Number of observation')
    plt.title('Distribution of ISD')
    plt.show()
def sim_stock_prices_ISD(simnum, sigma, mu, N, S0,T, alpha):
    dt = 1/N
    Stock_price_paths=np.zeros((simnum, (T*N)+1))
    # Generate stock prices
    Stock_price_paths[:, 0] = ISD(simnum, S0, alpha)
    for i in range(0,simnum,2):
        for j in range(1,(T*N)+1):
            z=np.random.normal(loc=0, scale=1)
            Stock_price_paths[i,j] = Stock_price_paths[i,j-1] * np.exp((mu - sigma ** 2 / 2) * dt + sigma *z*np.sqrt(dt))
            Stock_price_paths[i+1, j] = Stock_price_paths[i+1, j - 1] * np.exp((mu - sigma ** 2 / 2) * dt - sigma *z*np.sqrt(dt))
    return Stock_price_paths

def polynomial_basis(X, x0, degree):
    basis = [(X - x0)**j for j in range(1,degree+1)]
    return np.column_stack(basis)

def OLS_Greeks(simnum, sigma, mu, N, T, S0, K,alpha):
    disc_vector = d(mu, N, T)
    AMoption_price, exerciseboundary = lsm(simnum, sigma, mu, N, T, K-4, K)
    Stock_price_paths = sim_stock_prices_ISD(simnum, sigma, mu, N, S0,T, alpha)
    X = Stock_price_paths[:, 0]
    Stock_price_paths = np.array(Stock_price_paths) / K
    cashflow_matrix = np.zeros((simnum, (T * N) + 1))

    for j in range(1,N * T+1, 1):
        exercise = Stock_price_paths[:,j]<=exerciseboundary[j]
        Stock_price_paths[exercise,j+1:]=0
        Stock_price_paths[~exercise, j] = 0
    for j in range(N * T, 0, -1):
        Cashflow = Stock_price_paths[:,j]>0
        cashflow_matrix[Cashflow, j] = np.maximum(0, 1 - Stock_price_paths[Cashflow, j])

    cashflow_disc = np.matmul(K*cashflow_matrix[:, 1:(N * T) + 1], np.transpose(disc_vector))
    Y_OLS = cashflow_disc
    X_OLS =polynomial_basis(X, S0, 8)
    X_OLS = sm.add_constant(X_OLS)
    model_OSL = sm.OLS(Y_OLS, X_OLS).fit()
    Price = model_OSL.params[0]
    Delta = model_OSL.params[1]
    return Price, Delta, Y_OLS, X

def tabel2(simnum):
    Strikeprise=np.array((36,40,44))
    PriceLSMISD1 = np.zeros((20,3))
    DeltaLSMISD1 = np.zeros((20,3))
    for j in range(3):
        print(j)
        for i in range(20):
            Price, Delta = OLS_Greeks(100000, 0.2, 0.06, 50, 1, 40, Strikeprise[j],25)
            PriceLSMISD1[i,j]=Price
            DeltaLSMISD1[i,j]=Delta
    column_averagesPrice = np.mean(PriceLSMISD1, axis=0)
    column_std_devPrice = np.std(PriceLSMISD1, axis=0, ddof=1)
    column_averagesDelta = np.mean(DeltaLSMISD1, axis=0)
    column_std_devDelta = np.std(DeltaLSMISD1, axis=0, ddof=1)
    return


#ISD_pricepaths_plot(10000, 0.2, 0.06, 50, 40,1, 0.5)
#ISD_distribution_plot(100000, 40, 0.5)
params=OLS_Greeks(100000, 0.2, 0.06, 50, 1,36,40, 5)
print(params)

simnum = 100000
sigma = 0.2
mu = 0.06 #interest rate if under Q.
N = 50
T = 1
S0 = 40
K = 40
alpha=0.5


def plota1naiv():
    Price, Delta, Y_OLS, X = OLS_Greeks(100000, 0.2, 0.06, 50, 1, 40, 40,0.5)
    plt.subplots(figsize=(12, 8))
    sns.set(style="darkgrid")
    color = sns.color_palette("tab20c", 20)
    plt.plot(X, Y_OLS, 'o', markersize=1, color=color[5])
    plt.xlabel('Stock price')
    plt.ylabel('Discounted Payoff')
    plt.title('Alpha=0.5')
    plt.show()

#plot figure A.1
plt.plot(X, Y_OLS, 'o', markersize=1)
plt.show()