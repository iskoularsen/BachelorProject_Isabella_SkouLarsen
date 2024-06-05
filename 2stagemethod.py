import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
from regression import *

simnum=50000
sigma=0.2
mu=0.06
TimeToMaturity=10
S0=36
T=1
alpha=5
N0=10
N=50
K=40
#used to dispurse the initial stock price
def ISD(simnum, S0, alpha):
    U = np.zeros(simnum)
    U_2 = np.random.uniform(0, 1, (simnum // 2 ,))
    for i in range(simnum // 2):
        U[2 * i] = U_2[i]
        U[2 * i + 1] = U_2[i]
    K_isd = 2 * np.sin(np.arcsin(2 * U - 1)/3)
    X = S0 + alpha * K_isd
    return X

#uses the ISD X vector and the antothetick paths to simulate stockprices
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

def ISD_pricepaths_plot(simnum, sigma, mu, N, S0,T, alpha):
    Stock_price_paths = sim_stock_prices_ISD(simnum, sigma, mu, N, S0,T, alpha)
    dt = 1 / N
    X =  np.arange(0, T+dt, dt)
    plt.subplots(figsize=(12, 8))
    sns.set(style="darkgrid", palette="Greens_d")
    plt.plot(X,Stock_price_paths.T)
    plt.xlabel('Time')
    plt.ylabel('Stock price')
    plt.title('ISD Stock Price Paths')
    plt.xticks(np.arange(0, T + dt, 0.1))
    #plt.legend()
    plt.show()

def ISD_distribution_plot(simnum, S0, alpha):
    X = ISD(simnum, S0, alpha)
    plt.subplots(figsize=(12, 8))
    Greens_palette = sns.color_palette("Greens", 10)
    sns.set(style="darkgrid")
    #greys_palette = sns.color_palette("Greys", 10)
    plt.hist(X, bins=35, color=Greens_palette[7])
    plt.xlabel('Stock price')
    plt.ylabel('Number of observation')
    plt.title('Distribution of ISD')
    #plt.axvline(S0 + alpha, color=greys_palette[5], linestyle=':', linewidth=1,)
    #plt.axvline(S0 - alpha, color=greys_palette[6], linestyle=':', linewidth=1)
    x_min = np.floor(np.min(X))
    x_max = np.ceil(np.max(X))
    # Set x-axis to show every natural number
    plt.xticks(np.arange(x_min, x_max + 1, 1))
    plt.show()


def sim_stock_prices_ISD_2(simnum, sigma, mu, N, S0,T, alpha,N0):
    dt = 1/N0
    Stock_price_paths=np.zeros((simnum, (T*N)+1))
    # Generate stock prices
    Stock_price_paths[:, 0] = ISD(simnum, S0, alpha)
    for i in range(0,simnum,2):
        for j in range(1,(T*N)+1):
            z=np.random.normal(loc=0, scale=1)
            Stock_price_paths[i,j] = Stock_price_paths[i,j-1] * np.exp((mu - sigma ** 2 / 2) * dt + sigma *z*np.sqrt(dt))
            Stock_price_paths[i+1, j] = Stock_price_paths[i+1, j - 1] * np.exp((mu - sigma ** 2 / 2) * dt - sigma *z*np.sqrt(dt))
    return Stock_price_paths


#Function to use in the OSL regression, output ik the modelmatrix, without the intercept
def polynomial_basis(X, x0, degree):
    basis = [(X - x0)**j for j in range(1,degree+1)]
    return np.column_stack(basis)
def DeltafunkLSMISD(S0, K, T,N,mu,sigma, simnum,alpha):
    Stockprices = S0 + np.arange(-20, S0, 1)
    DeltaLSMISD =  np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        print(i)
        price, delta = twostage(simnum, sigma, mu, N, T, Stockprices[i], K,alpha)
        DeltaLSMISD[i]=delta
    return DeltaLSMISD

 def plotdeltaLSM(0, K, T,N,mu,sigma, simnum,alpha):
    Delta_LSM_ISD =DeltafunkLSMISD(S0, K, T,N,mu,sigma, simnum,alpha)
    plt.figure(figsize=(10, 6))
    sns.set(style="darkgrid")
    X = np.arange(16, S0 + S0, 1)
    color = sns.color_palette("tab20c", 20)
    colors = sns.color_palette("GnBu", 10)
    plt.plot(X, Deltabinom[36 - 20:36 + 36], '-', linewidth=3.0, label='BM', color=colors[7])
    plt.plot(X, DeltaLSMISD, 'o-', markersize=3, label='Alpha 5 ', color=color[5])
    plt.xlim((16, 70))
    plt.xlabel('Option Price')
    plt.ylabel('Delta')
    plt.title('Delta function - LSM ISD')
    plt.legend()
    plt.show()

#The two stage method output is the option price and greeks
def twostage(simnum, sigma, mu, N, T, S0, K,alpha):
    disc_vector = d(mu, N, T) #discount vector from Regression.py
    AMoption_price, exerciseboundary = lsm(simnum, sigma, mu, N, T, K-4, K) #from regression.py
    #exerciseboundary = exerciseboundary
    Stock_price_paths = sim_stock_prices_ISD(simnum, sigma, mu, N, S0,T, alpha)
    Stock_price_paths = np.array(Stock_price_paths) / K
    cashflow_matrix = np.zeros((simnum, (T * N) + 1))
    #using the exercise boundary obtained from LSM we determine from time 2 when it is optimal i exercise
    for j in range(2,N * T+1, 1):
        exercise = Stock_price_paths[:,j]<=exerciseboundary[j]
        Stock_price_paths[exercise,j+1:]=0
        Stock_price_paths[~exercise, j] = 0

    #Determine the cashflow genereatet from the optiomal strategy from time 2 and immideate payoff af time 1.
    # np.where(Stock_price_paths[:,j]>0, cashflow_matrix=1 - Stock_price_paths,0 )
    # np.where(Stock_price_paths[:, j] > 0, cashflow_matrix[:,j]= np.maximum(0, 1 - Stock_price_paths[:, j], 0)
    for j in range(N * T, 0, -1):
        Cashflow = Stock_price_paths[:,j]>0
        cashflow_matrix[Cashflow, j] = np.maximum(0, 1 - Stock_price_paths[Cashflow, j])

    #discounts the cashflows generated from the optimal stopping strategy to time 1.
    cashflow_disc = np.matmul(cashflow_matrix[:, 2:(N * T) + 1], np.transpose(disc_vector[0:(N * T) -1]))

    #Preforming the regression method from LSM to determine the continuation value for all paths (why?? idk)
    # dette skal være laguerre
    X_regression = Stock_price_paths[:, 1]
    Y_regression = cashflow_disc
    #model = np.polyfit(X_regression, Y_regression, 8)
    #Continuation = np.polyval(model, X_regression)
    Basis = laguerre_basis(X_regression)
    model = LinearRegression().fit(Basis,  Y_regression)
    Continuation = model.predict(Basis)

    #Determine the valuefunction at time 1 and discounting to time zero
    Vt1 = np.maximum(Continuation, cashflow_matrix[:, 1])
    Vt1_disc = Vt1 * disc_vector[0]

    #preforming the regression to determine the greeks
    Y_OLS = K*Vt1_disc
    X = K*Stock_price_paths[:, 0]
    X_OLS = polynomial_basis(X, S0, 8)
    X_OLS = sm.add_constant(X_OLS)
    model_OSL = sm.OLS(Y_OLS, X_OLS).fit()
    Price = model_OSL.params[0]
    Delta = model_OSL.params[1]
    return Price, Delta
    # Y_OLS, X

def plota1twostage():
    Price, Delta,Gamma, Y_OLS, X = twostage(100000, 0.2, 0.06, 50, 1, 40, 40,25)
    plt.subplots(figsize=(12, 8))
    sns.set(style="darkgrid")
    color = sns.color_palette("tab20c", 20)
    plt.plot(X, Y_OLS, 'o', markersize=1, color=color[5])
    plt.xlabel('Stock price')
    plt.ylabel('Discounted Payoff')
    plt.title('Alpha=25')
    #plt.ylim((0,10))
    plt.show()

def tabel2(simnum):
    Strikeprise=np.array((36,40,44))
    PriceLSMISD1 = np.zeros((50,3))
    DeltaLSMISD1 = np.zeros((50,3))
    for j in range(3):
        print(j)
        for i in range(50):
            Price, Delta, gamma = twostage(100000, 0.2, 0.06, 50, 1, 40, Strikeprise[j],25)
            PriceLSMISD1[i,j]=Price
            DeltaLSMISD1[i,j]=Delta
    column_averagesPrice = np.mean(PriceLSMISD1, axis=0)
    column_std_devPrice = np.std(PriceLSMISD1, axis=0, ddof=1)
    column_averagesDelta = np.mean(DeltaLSMISD1, axis=0)
    column_std_devDelta = np.std(DeltaLSMISD1, axis=0, ddof=1)
    return


#USED
#used to determine the cashflow_disc, Stock_price_paths, cashflow_matrix, disc_vector
def Cashflowtime(simnum, sigma, mu, TimeToMaturity, S0,T, alpha,N0,N,K,Exerciseboundary):
    disc_vector = d(mu, N0, T) #discount vector for the stockpricepath
    steps = N//N0 #steps
    exerciseboundary=Exerciseboundary/K
    #generates the stockpaths with length N from the IDS distribution
    Stock_price_paths = sim_stock_prices_ISD_2(simnum, sigma, mu, TimeToMaturity, S0,T, alpha,N0)
    Stock_price_paths = np.array(Stock_price_paths) / K #normalize the process
    cashflow_matrix = np.zeros((simnum, (T * TimeToMaturity) + 1)) #empty cashflow

    #this has to start one period after where we are
    for j in range(2, TimeToMaturity * T+1, 1):
        exercise = Stock_price_paths[:,j]<=exerciseboundary[j*steps+(N0-TimeToMaturity)*steps]
        Stock_price_paths[exercise,j+1:]=0
        Stock_price_paths[~exercise, j] = 0
    for j in range(TimeToMaturity * T, 0, -1):
        Cashflow = Stock_price_paths[:,j]>0
        cashflow_matrix[Cashflow, j] = np.maximum(0, 1 - Stock_price_paths[Cashflow, j])
    cashflow_disc = np.matmul(cashflow_matrix[:, 2:(TimeToMaturity  * T) + 1], np.transpose(disc_vector[0:(TimeToMaturity * T-1)]))
    return cashflow_disc, Stock_price_paths, cashflow_matrix, disc_vector

#USED
#first regression in two stagemethos
def regressionstep(simnum, sigma, mu, TimeToMaturity, S0,T, alpha,N0,N,K,Exerciseboundary):
    cashflow_disc, Stock_price_paths, cashflow_matrix , disc_vector = Cashflowtime(simnum, sigma, mu, TimeToMaturity, S0,T, alpha,N0,N,K,Exerciseboundary)
    # Preforming the regression method from LSM to determine the continuation value for all paths
    X_regression = Stock_price_paths[:, 1]
    Y_regression = cashflow_disc
    Basis = laguerre_basis(X_regression)
    model = LinearRegression().fit(Basis, Y_regression)
    Continuation = model.predict(Basis)
    # Determine the valuefunction at time 1 and discounting to time zero
    Vt1 = np.maximum(Continuation, cashflow_matrix[:, 1])
    Vt1_disc = Vt1 * disc_vector[0]
    # preforming the regression to determine the greeks
    Y_OLS = K * Vt1_disc
    X = K * Stock_price_paths[:, 0]
    X_OLS = polynomial_basis(X, S0, 8)
    X_OLS = sm.add_constant(X_OLS)
    model_OSL = sm.OLS(Y_OLS, X_OLS).fit()
    Price = model_OSL.params[0]
    Delta = model_OSL.params[1]
    return Price, Delta

#USED
#HEDGIG ERROR FOR FRIST LSM ISD METHOD
def DeltaLSMIDS(Exerciseboundary, StockpricesDelta, N0,):
    Hedgingtime = hedgetime2(StockpricesDelta, Exerciseboundary, N2, N0)
    HedgingerorIS = np.full(len(StockpricesDelta), np.nan)

    start = time.time()
    for j in range(800,1000):
        print('stock path', j)
        Hedgingtimej = int(Hedgingtime[j])
        print('Hedgingtime', Hedgingtimej)
        StockpricesDeltaj = StockpricesDelta[j]
        price = np.zeros(Hedgingtimej)
        delta = np.zeros(Hedgingtimej)
        for i in range(0,Hedgingtimej):
            print(i)
            Price, Delta = regressionstep(100000, sigma, mu, N0-i, StockpricesDeltaj[i], T, alpha, N0, N, K,Exerciseboundary)
            price[i]=Price
            delta[i]=Delta
        delta = np.clip(delta, -1, 0)
        Cash = np.zeros(Hedgingtimej)
        RP = np.zeros(Hedgingtimej + 1)
        RP[0] = 4.4756271241618295
        disc = np.exp(mu * (1 / N0))
        Cash[0] = RP[0] - delta[0] * StockpricesDeltaj[0]
        for i in range(1, Hedgingtimej):
            RP[i] = Cash[i - 1] * disc + delta[i - 1] * StockpricesDeltaj[i]
            Cash[i] = RP[i] - delta[i] * StockpricesDeltaj[i]
        RP[Hedgingtimej] = Cash[Hedgingtimej - 1] * disc + delta[Hedgingtimej - 1] * StockpricesDeltaj[Hedgingtimej]
        hedgerror = np.maximum(0, K - StockpricesDeltaj[Hedgingtimej]) - RP[Hedgingtimej]
        timetomaturity = (N0 - Hedgingtimej) / N0
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        HedgingerorISD[j] = dischedgerror
    end = time.time()
    print(end - start)
    return HedgingerorISD

def plotdeltadistLSMISD():
    Hedgingerror1, HedgingerrorT=DeltaBlackSchoels2(T,N,sigma,mu,K, Exerciseboundary,Stockpricestest,N2)
    plt.subplots(figsize=(10, 6))
    sns.set(style="darkgrid")
    color = sns.color_palette("tab20c", 20)
    plt.hist(HedgingerorISD, bins=20, label='Hedging Error',  color=color[5], density=True)
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.title('Hedging error distribution - LSM ISD')
    plt.show()

#USED TO DETERMINE THE COEF IN THE DELTA FUNKTION METHOD
def deltaLSMISD2(Exerciseboundary, simnum, sigma, mu, N, S0,T, alpha, K):
    stockprices=sim_stock_prices_ISD(simnum, sigma, mu, N, S0,T, alpha)
    disc_vector = d(mu, N, T)
    # using the exercise boundary obtained from LSM we determine from time 2 when it is optimal i exercise
    coef = np.zeros((N, 9))
    for i in range(0,N):
        Stockpricesoptim = np.array(stockprices)
        cashflow_matrix = np.zeros((simnum, (T * N) + 1))
        for j in range(i+2, N * T + 1, 1):
            exercise = Stockpricesoptim[:, j] <= Exerciseboundary[j]
            Stockpricesoptim[exercise, j + 1:] = 0
            Stockpricesoptim[~exercise, j] = 0

            # Determine the cashflow genereatet from the optiomal strategy from time 2 and immideate payoff af time 1.
        for j in range(N * T, i, -1):
            Cashflow = Stockpricesoptim[:, j] > 0
            cashflow_matrix[Cashflow, j] = np.maximum(0, K - Stockpricesoptim[Cashflow, j])

        cashflow_disc = np.matmul(cashflow_matrix[:, i+2:(N * T) + 1], np.transpose(disc_vector[0:(N * T) - (i + 1)]))

        X1 = stockprices[:, i + 1]
        Basis = laguerre_basis(X1)
        Y_regression = cashflow_disc
        model = LinearRegression().fit(Basis, Y_regression)
        Continuation= model.predict(Basis)

    # Determine the valuefunction at time 1 and discounting to time zero
        Vt1 = np.maximum(Continuation, cashflow_matrix[:, 1+i])
        Vt1_disc = Vt1 * disc_vector[0]

        # preforming the regression to determine the greeks
        Y_OLS =  Vt1_disc
        X = stockprices[:,i]
        X_OLS = polynomial_basis(X, S0, 8)
        X_OLS = sm.add_constant(X_OLS)
        model_OSL = sm.OLS(Y_OLS, X_OLS).fit()
        coef[i,:] =  model_OSL.params
    return coef


#USED IN THE DELTA FUNKTION (DeltafunktiongenLSMISD)
def deltafunction2(Stockprice, S0,coef, Hedgingtime):
    delta =  np.full(len(Stockprice), np.nan)
    for i in range(Hedgingtime):
        delta[i] = coef[i, 1] + 2 * coef[i, 2] * (Stockprice[i] - S0) + 3 * coef[i, 3] * (Stockprice[i] - S0) ** 2 + 4 * coef[i, 4] * (Stockprice[i] - S0) ** 3 + 5 * coef[i, 5] * (Stockprice[i] - S0) ** 4 + 6 * coef[i, 6] * (Stockprice[i] - S0) ** 5 + 7 * coef[i, 7] * (Stockprice[i] - S0) ** 6 + 8 * coef[i, 8] * (Stockprice[i] - S0) ** 7
    Delta = np.clip(delta, -1, 0)
    return Delta

#HEDGING ERROR USING THE DELTA FUNKTION METHOD IN LSM ISD
def DeltafunktiongenLSMISD(Exerciseboundary, StockpricesDelta, N2, N0,simnum, sigma, mu, N, S0,T, alpha, K ):
    coef = deltaLSMISD2( Exerciseboundary, simnum, sigma, mu, N, S0,T, alpha, K)
    Hedgingtime = hedgetime2(StockpricesDelta, Exerciseboundary, N2, N0)
    HedgingerorISD2 = np.full(len(StockpricesDelta), np.nan)
    for j in range(len(StockpricesDelta)):
        Hedgingtimej = int(Hedgingtime[j])
        StockpricesDeltaj = StockpricesDelta[j]
        Delta=deltafunction2(StockpricesDeltaj, S0,coef, Hedgingtimej)
        Cash = np.zeros(Hedgingtimej)
        RP = np.zeros(Hedgingtimej + 1)
        RP[0] = 4.4756271241618295
        disc = np.exp(mu * (1 / N0))
        Cash[0] = RP[0] - Delta[0] * StockpricesDeltaj[0]
        for i in range(1, Hedgingtimej):
            RP[i] = Cash[i - 1] * disc + Delta[i - 1] * StockpricesDeltaj[i]
            Cash[i] = RP[i] - Delta[i] * StockpricesDeltaj[i]
        RP[Hedgingtimej] = Cash[Hedgingtimej - 1] * disc + Delta[Hedgingtimej - 1] * StockpricesDeltaj[Hedgingtimej]
        hedgerror = np.maximum(0, K - StockpricesDeltaj[Hedgingtimej]) - RP[Hedgingtimej]
        timetomaturity = (N0 - Hedgingtimej) / N0
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        HedgingerorISD2[j] = dischedgerror
    return HedgingerorISD2

def plotdeltadistLSMISD2():
    Hedgingerror1, HedgingerrorT=DeltaBlackSchoels2(T,N,sigma,mu,K, Exerciseboundary,Stockpricestest,N2)
    plt.subplots(figsize=(10, 6))
    sns.set(style="darkgrid")
    color = sns.color_palette("tab20c", 20)
    plt.hist(HedgingerorISD2, bins=20, alpha=0.75, label='Hedging Error',  color=color[5], density=True)
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Hedging error distribution - LSM ISD Method Two')
    plt.show()


#DELTAFUNKTION AF LSM ISD DELTA FUNKTION METODEN
def DeltafunkLSMISD2(S0, K, T,N,r,sigma):
    Stockprices = S0 + np.arange(-20, S0, 1)
    DeltafunkISD2 =  np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        DeltafunkISD2[i] = coef[0, 1] + 2 * coef[0, 2] * (Stockprices[i] - S0) + 3 * coef[0, 3] * (
                    Stockprices[i] - S0) ** 2 + 4 * coef[0, 4] * (Stockprices[i] - S0) ** 3 + 5 * coef[0, 5] * (
                               Stockprices[i] - S0) ** 4 + 6 * coef[0, 6] * (Stockprices[i] - S0) ** 5 + 7 * coef[
                       0, 7] * (Stockprices[i] - S0) ** 6 + 8 * coef[0, 8] * (Stockprices[i] - S0) ** 7
    DeltafunkISD2 = np.clip(DeltafunkISD2, -1, 0)
    return DeltafunkISD2

 def plotdeltaLSM2(0, K, T,N,mu,sigma, simnum,alpha):
    Delta_LSM_ISD =DeltafunkLSMISD(S0, K, T,N,mu,sigma, simnum,alpha)
    plt.figure(figsize=(10, 6))
    sns.set(style="darkgrid")
    X = np.arange(16, S0 + S0, 1)
    color = sns.color_palette("tab20c", 20)
    colors = sns.color_palette("GnBu", 10)
    plt.plot(X, Deltabinom[36 - 20:36 + 36], '-', linewidth=3.0, label='BM', color=colors[7])
    plt.plot(X, DeltafunkISD2, 'o-', markersize=3, label='Alpha 5 ', color=color[5])
    plt.xlim((16, 70))
    plt.xlabel('Option Price')
    plt.ylabel('Delta')
    plt.title('Delta function - LSM ISD (Method Two)')
    plt.legend()
    plt.show()
#NOT USED
def inputs2stage():
    AMoption_price, exerciseboundary = lsm(simnum, sigma, mu, N, T, K - 4, K)
    Stock_price_paths = sim_stock_prices(simnum, sigma, mu, N, S0, T, alpha)
    Stock_price_paths = np.array(Stock_price_paths) / K
    return Stock_price_paths, exerciseboundary

#NOT USED
def hedgetime(exerciseboundary, N2, N0, sigma, mu, T, S0, K):
    exercisesteps = N2 // N0
    optionalive = np.zeros(N0 - 1)
    # AMoption_price, exerciseboundary = lsm(100000, sigma, mu, N2, T, S0, K)  # determines the boundary
    Exerciseboundary = exerciseboundary * K
    Stockprice = sim_stock_prices2(1, sigma, mu, N0, T, S0)  # from Brwonian_motion.py
    for i in range(1, N0):  # determines number of rebalacing points where the option is alive
        if Stockprice[0, i] > Exerciseboundary[i * exercisesteps]:
            optionalive[i - 1] = 1
        else:
            break
    HedgeTime = int(sum(optionalive)) + 1  # the time we need to delta hedge
    return HedgeTime, Exerciseboundary, Stockprice

#NOT USED
def DeltaLSM_ISD(HedgeTime, Stockprice, exerciseboundary,simnum, sigma, mu,T,alpha,N0,N,K):
    price = np.zeros(HedgeTime)
    delta = np.zeros(HedgeTime)
    for i in range(0,HedgeTime):
        Price, Delta = regressionstep(simnum, sigma, mu, N0-i, Stockprice[0,i], T, alpha, N0, N, K,exerciseboundary)
        price[i]=Price
        delta[i]=Delta
    Cash = np.zeros(HedgeTime)
    RP = np.zeros(HedgeTime + 1)
    RP[0] = price[0]
    disc = np.exp(mu * (1 / N0))
    Cash[0] = price[0] - delta[0] * Stockprice[0, 0]
    for i in range(1, HedgeTime):
        RP[i] = Cash[i - 1] * disc + delta[i - 1] * Stockprice[0, i]
        Cash[i] = RP[i] - delta[i] * Stockprice[0, i]
    RP[HedgeTime] = Cash[HedgeTime - 1] * disc + delta[HedgeTime - 1] * Stockprice[0, HedgeTime]
    price = np.append(price, np.maximum(0, K - Stockprice[0, HedgeTime]))
    hedgerror = RP[HedgeTime] - price[HedgeTime]
    timetomaturity = (N0 - HedgeTime) / N0
    dischedgerror = hedgerror * np.exp(mu * timetomaturity)
    #FROM THE OTHER
    Cash = np.zeros(Hedgingtimej)
    RP = np.zeros(Hedgingtimej + 1)
    RP[0] = price[0]
    disc = np.exp(mu * (1 / N0))
    Cash[0] = price[0] - delta[0] * Stockprice[0, 0]

    for i in range(1, HedgeTime):
        RP[i] = Cash[i - 1] * disc + delta[i - 1] * Stockprice[0, i]
        Cash[i] = RP[i] - delta[i] * Stockprice[0, i]
    RP[HedgeTime] = Cash[Hedgingtimej - 1] * disc + delta[Hedgingtimej - 1] * Stockprice[0, HedgeTime]
    price = np.append(price, np.maximum(0, K - Stockprice[0, HedgeTime]))
    hedgerror = RP[HedgeTime] - price[HedgeTime]
    timetomaturity = (N0 - HedgeTime) / N0
    dischedgerror = hedgerror * np.exp(mu * timetomaturity)
    HedgingerorISD[j] = dischedgerror
    return dischedgerror

#MULTI DELTA FUNCTION PLOT
 def plotdeltaLSM(Stockprices,Exerciseboundary,epsilon, simnum, sigma, mu,T,K):
    plt.figure(figsize=(10, 6))
    sns.set(style="darkgrid")
    X = np.arange(16, S0 + S0, 1)
    colors_custom = [
        (0.9, 0.675, 0.9),  # Light Pink
        "#800080",  # Purple
        "#FF69B4"  # Hot Pink
    ]
    colors2 = sns.color_palette("Purples_d", 6)
    colors = sns.color_palette("GnBu", 10)
    colors1 = sns.color_palette("gist_yarg", 6)
    color3 = sns.color_palette("tab20c", 20)
    plt.plot(X, Deltabinom[36 - 20:36 + 36], '-', linewidth=3.0, label='BM', color=colors[7])
    plt.plot(X, DeltaLSM001, '-', markersize=3, label='LSM', color=colors_custom[1])
    plt.plot(Delta1, '-', markersize=3, label='BS', color=colors1[5])
    plt.plot(X, DeltafunkISD2, '-', markersize=3, label='LSM ISD 2 ', color=color3[5])
    plt.plot(X, DeltaLSMISD, '-', markersize=3, label='LSM ISD ', color=color3[7])
    plt.xlim((16, 70))
    plt.xlabel('Option Price')
    plt.ylabel('Delta')
    plt.title('Delta functions')
    plt.legend(['BM', 'LSM', 'BS', 'LSM ISD 2', 'LSM ISD'], loc='upper left', bbox_to_anchor=(0, 1), title="Model")
    plt.show()



#Different deltahedging error distributions
Delta1binom #binom
Hedgingerror #Black Scholes
DeltahedgerrorLSM #LSM
HedgingerorISD2 #LSM ISD Method Two

a = pd.DataFrame(Delta1binom)
b = pd.DataFrame(Hedgingerror)
c = pd.DataFrame(DeltahedgerrorLSM)
e = pd.DataFrame(HedgingerorISD2)
df = a.append(b).append(c).append(e).append(e)

# Usual boxplot
a = pd.DataFrame({'group' : np.repeat('BM',1000), 'value': Delta1binom})
b = pd.DataFrame({'group' : np.repeat('Black-Scholes',1000), 'value': Hedgingerror})
c = pd.DataFrame({'group' : np.repeat('LSM',1000), 'value': Hedgingerorbackup})
f = pd.DataFrame({'group' : np.repeat('LSM ISD',1000), 'value': HedgingerorISD})
e = pd.DataFrame({'group' : np.repeat('LSM ISD Method Two',1000), 'value': HedgingerorISD2})
# Combine all DataFrames using pd.concat and reset the index
df = pd.concat([a, b, c, f, e]).reset_index(drop=True)
colors_custom = [
        (0.9, 0.675, 0.9),  # Light Pink
        "#800080",  # Purple
        "#FF69B4"  # Hot Pink
    ]
    colors2 = sns.color_palette("Purples_d", 6)
    colors = sns.color_palette("GnBu", 10)
    colors1 = sns.color_palette("gist_yarg", 6)
    color3 = sns.color_palette("tab20c", 20)
# Define a custom color palette that matches your group labels
palette = {
    "BM": color3[0],
    "Black-Scholes": color3[16],
    "LSM": color3[12],
    "LSM ISD": color3[4],
    "LSM ISD Method Two": color3[5]
}
# Create the boxplot with custom colors
sns.boxplot(x='group', y='value', hue='group', data=df, palette=palette, dodge=False)
plt.xlabel('')
plt.xticks(rotation=0)  # Rotate labels if they overlap
plt.show()
