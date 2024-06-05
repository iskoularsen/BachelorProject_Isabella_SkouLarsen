import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import time
from numpy.polynomial.laguerre import lagfit, lagval
from Brownian_motion import *

simnum = 250000
sigma = 0.2
mu = 0.06
N = 6
T = 1
S0 = 36
K = 40
N2=50
epsilon=0.01
N0=10


#Discount vector, mu: Risk free rate, N: number of timesteps, T: time to maturity in years
def d(mu, N,T):
    dt = 1 / N
    t = np.concatenate([np.arange(dt, T, dt), [T]])
    disc_vector = np.exp(-mu * t)
    return disc_vector

#function to use in regression
def laguerre_basis(x):
    basis_matrix = np.zeros((len(x), 3))
    basis_matrix[:, 0] = np.exp(-x/2)
    basis_matrix[:, 1] = np.exp(-x/2)*(1-x)
    basis_matrix[:, 2] = np.exp(-x/2)*(1 - 2*x + x**2/2)
    return basis_matrix

#the LSM algorithm.
#simnum: number of simulation, sigma: volatillity, mu: short interest rate, N: number of excerise points,
# T: time to maturity, S0: initial stock price, K: strikprice
def lsm(simnum,sigma,mu,N,T,S0, K):
    disc_vector =d(mu, N,T)
    Stock_price_paths = sim_stock_prices(simnum, sigma, mu, N, T, S0) #from brownian_motion
    Stock_price_paths=np.array(Stock_price_paths)/K #Normalize
    cashflow_matrix= np.zeros((simnum,(T*N)+1))
    exerciseboundary = np.full((T * N + 1,), np.NaN)
    #initial cashflow matrix
    for j in range(N*T, 0, -1):
        cashflow_matrix[:, j] = np.maximum(0, 1 - Stock_price_paths[:, j])
    #recrusive regression, with only ITM paths for each period, starting at N*T-1
    for k in range(N*T, 1, -1):
        ITM = 1 - Stock_price_paths[:, k-1] > 0
        X = Stock_price_paths[ITM, k - 1] # ITM stockprices to use in regression
        Y = np.matmul(cashflow_matrix[ITM, k:(N * T) + 1], np.transpose(disc_vector[:(N * T) - (k - 1)])) #ITM cashflows

        #Regression using polynomial basis function
        #model = np.polyfit(X, Y, 9)
        #Continuation = np.zeros((simnum,))
        #Continuation[ITM] = np.polyval(model, X)

        #Regression unsing polynomial basis function
        Basis = laguerre_basis(X)
        model = LinearRegression().fit(Basis, Y)
        Continuation = np.zeros((simnum,))
        Continuation[ITM] = model.predict(Basis)

        #Comparing the contionuation value with immediate payoff for ITM paths, i.e following optim stopping strategy
        optimhold = ITM & (Continuation <= cashflow_matrix[:, k-1])
        cashflow_matrix[optimhold, k:(N * T) + 1] = 0
        cashflow_matrix[~optimhold, k - 1] = 0

    #Determaning the exerciseboundary by finding every entry with a positive cashflow.
    Exercise = np.where(cashflow_matrix>0, Stock_price_paths,np.nan)
    for i in range(N * T, 0, -1):
        exerciseboundary[i] = np.nanmax(Exercise[:,i])

    #Discounting the cashflows back till time zero
    cashflow_disc=np.matmul(cashflow_matrix[:,1:(N*T)+1],np.transpose(disc_vector[0:N*T]))

    #Calculates the option price
    AMoption_price=K*np.sum(cashflow_disc)/simnum
    return AMoption_price, exerciseboundary

AMoption_price, exerciseboundary=lsm(100000,0.2,0.06,500,1,36, 40)

#Determines the price given differen strike prises
def tabel3(simnum):
    Strikeprise=np.array((36,40,44))
    PriceLSM = np.zeros((50,3))
    for j in range(3):
        print(j)
        for i in range(50):
            AMoption_price, exerciseboundary=lsm(simnum, 0.2, 0.06, 50, 1, 40, Strikeprise[j])
            PriceLSM[i,j]=AMoption_price
    column_averagesPrice = np.mean(PriceLSM, axis=0)
    column_std_devPrice = np.std(PriceLSM, axis=0, ddof=1)
    return column_averagesPrice, column_std_devPrice

#used to determine the average price given different timesteps
def tabel2(simnum):
    timestep=np.array((10,50,100,500))
    LSM1 = np.zeros((20, 4))
    for j in range(4):
        print(j)
        for i in range(20):
            AMoption_price, exerciseboundary = lsm(simnum, 0.2, 0.06, timestep[j], 1, 36, 40)
            LSM1[i,j]=AMoption_price
    column_averages = np.mean(LSM1, axis=0)
    column_std_dev = np.std(LSM1, axis=0, ddof=1)
    standard_errors = column_std_dev / np.sqrt(LSM1.shape[0])
    return standard_errors, column_averages

#determiens option price for differen values of exercise points
def tabel3():
    optionprice50000 = np.full(502, np.nan)
    for i in range(2, 502,2):
        print(i)
        AMoption_price, exerciseboundary = lsm(50000, 0.2, 0.06, i, 1, 36, 40)
        optionprice50000[i] = AMoption_price


optionpricebinom1=optionpricebinom[~np.isnan(optionpricebinom)]
optionprice500001= optionprice50000[~np.isnan(optionprice50000)]
optionprices12 = optionprice[~np.isnan(optionprice)]
optionprice1000001=optionprice100000[~np.isnan(optionprice100000)]

# Plots the LSM option price as a funktion og exercise points.
def plotcovLSM():
    plt.figure(figsize=(12, 6))
    sns.set(style="darkgrid")
    colors_custom = [
        (0.9, 0.675, 0.9),  # Light Pink
        "#800080",  # Purple
        "#FF69B4"  # Hot Pink
    ]
    colors2 = sns.color_palette("GnBu", 10)
    X = np.arange(2, 502, 2)
    X1 = np.arange(0, 502, 1)
    plt.plot(X1, optionpricebinom, linestyle='-', label='BM', color=colors2[7])
    plt.plot(X, optionprices12, linestyle='-', label='LSM 25,000',  color=colors_custom[0])
    plt.plot(X, optionprice500001, linestyle='-', label='LSM 50,000',  color=colors_custom[2])
    plt.plot(X, optionprice1000001, linestyle='-', label='LSM 100,000',  color=colors_custom[1])
    plt.xlabel('Exercise Points')
    plt.ylabel('Stock price')
    plt.title("Convergence of the LSM Algorithm")
    plt.legend()
    plt.show()

#plots boundart given different number of time steps
def LSMboudaryplot1(K, S0, T, r, sigma):
    AMoption_price, exerciseboundary50 = lsm(100000, 0.2, 0.06, 50, 1, 36, 40)
    AMoption_price, exerciseboundary100 = lsm(100000, 0.2, 0.06, 100, 1, 36, 40)
    AMoption_price, exerciseboundary500 = lsm(100000, 0.2, 0.06, 500, 1, 36, 40)
    dt1 = 1 / 50
    dt2 = 1 / 100
    dt4 = 1 / 500
    X2 = np.arange(0, T +dt2, dt2)
    X= np.arange(0, T + dt1, dt1)
    X4 = np.arange(0, T + dt4, dt4)
    dt3 = 1 / 10000
    X3 = np.arange(dt3, T + dt3, dt3)
    plt.subplots(figsize=(12, 8))
    sns.set(style="darkgrid")
    colors_custom = [
        (0.9, 0.675, 0.9),  # Light Pink
        "#800080",  # Purple
        "#FF69B4"  # Hot Pink
    ]
    colors2 = sns.color_palette("GnBu", 10)
    plt.plot(X3, Exerciseboundary10000, label='Boundary 10000 steps', color=colors2[7])
    plt.plot(X4, exerciseboundary500 * K, label='Boundary 10000 steps', color=colors_custom[1])
    plt.plot(X2, exerciseboundary100 * K, label='Boundary 1000 steps', color=colors_custom[2])
    plt.plot(X, exerciseboundary50*K, label='Boundary 50 steps',color=colors_custom[0])
    plt.xlabel('Time')
    plt.ylabel('Stock price')
    plt.title('Exercise boundary - LSM')
    plt.legend(['BM', 'J=500','J=100' , 'J=50'], loc='upper left', bbox_to_anchor=(0, 1), title="Steps")
    plt.show()

#plots exerciseboundart given different simulation numbers
def LSMboudaryplot3(K, S0, T, r, sigma):
    AMoption_price, exerciseboundaryp3 = lsm(100000, 0.2, 0.06, 50, 1, 36, 40)
    AMoption_price, exerciseboundarypl3 = lsm(50000, 0.2, 0.06, 50, 1, 36, 40)
    AMoption_price, exerciseboundaryplt3 = lsm(25000, 0.2, 0.06, 50, 1, 36, 40)
    dt1 = 1 / 50
    X= np.arange(0, T + dt1, dt1)
    dt3 = 1 / 10000
    X3 = np.arange(dt3, T + dt3, dt3)
    plt.subplots(figsize=(12, 8))
    sns.set(style="darkgrid")
    colors_custom = [
        (0.9, 0.675, 0.9),  # Light Pink
        "#800080",  # Purple
        "#FF69B4"  # Hot Pink
    ]
    colors2 = sns.color_palette("GnBu", 10)
    plt.plot(X3, Exerciseboundary10000, label='BM', color=colors2[7])
    plt.plot(X, exerciseboundaryplt3*K, label='25000',color=colors_custom[0])
    plt.plot(X, exerciseboundarypl3 * K, label='50000', color=colors_custom[2])
    plt.plot(X, exerciseboundaryp3 * K, label='100000', color=colors_custom[1])
    plt.xlabel('Time')
    plt.ylabel('Stock price')
    plt.title('Exercise boundary - LSM')
    plt.legend(['BM', '25,000','50,000' , '100,000'], loc='upper left', bbox_to_anchor=(0, 1), title="Simulations")
    plt.show()

#Use the plot the exercise boundary given differen inital stock pices
def LSMboudaryplot2(K, S0, T, r, sigma):
    AMoption_price, exerciseboundary36 = lsm(100000, 0.2, 0.06, 50, 1, 36, 40)
    AMoption_price, exerciseboundary40 = lsm(100000, 0.2, 0.06, 50, 1, 40, 40)
    AMoption_price, exerciseboundary44 = lsm(100000, 0.2, 0.06, 50, 1, 44, 40)
    AMoption_price, exerciseboundary32 = lsm(100000, 0.2, 0.06, 50, 1, 32, 40)
    dt1 = 1 / 50
    X= np.arange(0, T + dt1, dt1)
    dt3 = 1 / 10000
    X3 = np.arange(dt3, T + dt3, dt3)
    plt.subplots(figsize=(12, 8))
    sns.set(style="darkgrid")
    colors2 = sns.color_palette("GnBu", 10)
    colors_custom = [
        (0.9, 0.675, 0.9),  # Light Pink
        "#800080",  # Purple
        "#FF69B4"  # Hot Pink
    ]
    #plt.plot(X, exerciseboundary36*K, label='Boundary 100,000 sim',color=colors[5] )
    plt.plot(X3, Exerciseboundary10000, label='BM', color=colors2[7])
    plt.plot(X, exerciseboundary32 * K, label='S032', color=colors_custom[0])
    plt.plot(X,  exerciseboundary40*K, label='S040', color=colors_custom[2])
    plt.plot(X, exerciseboundary44*K, label='S044',color=colors_custom[1])
    plt.xlabel('Time')
    plt.ylabel('Stock price')
    plt.title('Exercise boundary - The Binomial model')
    plt.legend(['BM','S0=32', 'S0=40', 'S0=44'], loc='upper left', bbox_to_anchor=(0, 1), title="Initial Stock Price")
    plt.show()

#for table 1
def table1lsm():
    for i in range(30):
        AMoption_price, exerciseboundary=lsm(simnum, 0.2, mu, N, 1, S0, K)
        LSMprice17[i+20]=AMoption_price
        AMoption_pric, exerciseboundary = lsm(simnum, 0.2, mu, N, 2, S0, K)
        LSMprice18[i+20] = AMoption_pric
        AMoption_pri, exerciseboundary = lsm(simnum, 0.4, mu, N, 1, S0, K)
        LSMprice19[i+20] = AMoption_pri
        AMoption_pr, exerciseboundary = lsm(simnum, 0.4, mu, N, 2, S0, K)
        LSMprice20[i+20] = AMoption_pri
        print(i)
    return LSMprice17, LSMprice18, LSMprice19, LSMprice20


#NOT USED
def DeltaLSM(S0, K, T,N,r,sigma):
    Stockprices = S0 + np.arange(-S0, S0, 1)
    Delta =  np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        esp =Stockprices[i-1]*0.01
        AMoption_price1, exerciseboundary = lsm(50000, 0.2, 0.06, 50, 1, Stockprices[i-1], 40)
        AMoption_price2, exerciseboundary = lsm(50000, 0.2, 0.06, 50, 1, Stockprices[i-1]+eps, 40)
        Delta[i-1]=( AMoption_price2-AMoption_price1)/esp
    return Delta


#NOT USED
def Plotdeltesekantlsm():
    AMoption_price, exerciseboundary = lsm(100000, sigma, mu, 50, T, 36, 40)
    Exerciseboundary = exerciseboundary*K
    disc_vector = d(mu, N, T)
    Stockprices = S0 + np.arange(-20, 40, 1)
    LSM = np.zeros(len(Stockprices))
    LSMepsilon = np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        print(i)
        cashflow_matrix = np.zeros((50000, (T * N) + 1))
        cashflow_matrixeps = np.zeros((50000, (T * N) + 1))
        Stock_price_paths = sim_stock_prices(50000, sigma, mu, N, T, Stockprices[i-1])  # from brownian_motion
        #Stock_price_paths= np.array(Stock_price_paths) / K
        Stock_price_pathseps = sim_stock_prices(50000, sigma, mu, N, T,  Stockprices[i - 1]+0.01)  # from brownian_motion
        #Stock_price_pathseps = np.array(Stock_price_pathseps) / K
        for j in range(1, N * T + 1, 1):
            exercise = Stock_price_paths[:, j] <= Exerciseboundary[j]
            Stock_price_paths[exercise, j + 1:] = 0
            Stock_price_paths[~exercise, j] = 0
            exerciseeps = Stock_price_pathseps[:, j] <= Exerciseboundary[j]
            Stock_price_pathseps[exerciseeps, j + 1:] = 0
            Stock_price_pathseps[~exerciseeps, j] = 0
        for j in range(N * T, 0, -1):
            Cashflow = Stock_price_paths[:,j]>0
            cashflow_matrix[Cashflow, j] = K - Stock_price_paths[Cashflow, j]
            Cashfloweps = Stock_price_pathseps[:, j] > 0
            cashflow_matrixeps[Cashfloweps, j] = K - Stock_price_pathseps[Cashfloweps, j]
        cashflow_disc = np.matmul(cashflow_matrix[:, 1:(N* T) + 1],np.transpose(disc_vector[0:N* T]))
        LSM[i] = np.sum(cashflow_disc) / 50000
        cashflow_disceps = np.matmul(cashflow_matrixeps[:, 1:(N * T) + 1], np.transpose(disc_vector[0:N * T]))
        LSMepsilon[i] = np.sum(cashflow_disceps) / 50000
    Delta=(LSMepsilon - LSM)/0.01
    return Delta


#calculates the delta zero given a stock price
def Price2(Stockprices,Exerciseboundary,epsilon, simnum, sigma, mu,T,K,N):
    disc_vector = d(mu, N, T)
    Price = np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        print(i)
        Eps = Stockprices[i] * epsilon
        Stock_price_paths = sim_stock_prices(100000, sigma, mu, N, T, Stockprices[i]+Eps )
        cashflow_matrix = np.zeros((simnum, (T * N) + 1))
        for j in range(1, N* T+1, 1):
            exercise = Stock_price_paths[:,j]<=Exerciseboundary[j]
            Stock_price_paths[exercise,j+1:]=0
            Stock_price_paths[~exercise, j] = 0
            #for j in range(M * T, -1, -1):
        for j in range(N * T, 0, -1):
            Cashflow = Stock_price_paths[:,j]>0
            cashflow_matrix[Cashflow, j] = K - Stock_price_paths[Cashflow, j]
        cashflow_disc = np.matmul(cashflow_matrix[:, 1:(N * T) + 1], np.transpose(disc_vector[0:N * T ]))
        Price[i] = np.sum(cashflow_disc) / simnum
    return Price


#determines the Delta for only time zero. uses Price2 to determine delta
def DeltaLSM( Stockprices,Exerciseboundary,epsilon, simnum, sigma, mu,T,K):
    Stockprices = S0 + np.arange(-20, S0, 1)
    LSMeps=Price2(Stockprices,Exerciseboundary,epsilon, simnum, sigma, mu,T,K)
    LSM=Price2(Stockprices,Exerciseboundary,0, simnum, sigma, mu,T,K)
    Eps = Stockprices * epsilon
    Delta = (LSMeps - LSM) / Eps
    return Delta


#plots the delta funktion using the sektant method
 def plotdeltaLSM(Stockprices,Exerciseboundary,epsilon, simnum, sigma, mu,T,K):
    DeltaLSM=DeltaLSM(Stockprices,Exerciseboundary,epsilon, simnum, sigma, mu,T,K)
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
    plt.plot(X, Deltabinom[36 - 20:36 + 36], '-', linewidth=3.0, label='BM', color=colors[7])
    plt.plot(X, DeltaLSM05, 'o-', markersize=3, label='0.5', color='#9966CC')
    plt.plot(X, DeltaLSM01, 'o-', markersize=3, label='Delta 0.1', color= colors2[2])
    plt.plot(X, DeltaLSM001, 'o-', markersize=3, label='Delta 0.01', color=colors_custom[1])
    plt.plot(X, DeltaLSM0001, 'o-', markersize=3, label='Delta 0.001', color='#DA70D6')
    plt.xlim((16, 70))
    plt.xlabel('Option Price')
    plt.ylabel('Delta')
    plt.title('Delta function - LSM')
    plt.legend(['BM', '0.5', '0.1', '0.01', '0.001'], loc='upper left', bbox_to_anchor=(0, 1), title="Lambda")
    plt.show()


#NOT USED
#N2 exercisepoint to estimate the boundary
#N rebalacing points
def sekantlsm(simnum,sigma,mu,N,T,S0,K,N0,epsilon,N2):
    AMoption_price, exerciseboundary = lsm(100000, sigma, mu, N2, T, S0, K)
    Exerciseboundary = exerciseboundary*K
    Stockprice =  StockpricesDelta[9]
    LSM = np.zeros(N)
    LSMepsilon = np.zeros(N)

    disc_vector = d(mu, N0, T)

    for i in range(0,N):
        TimeToMaturity=N-i #time to maturity
        Stock_price_paths = sim_stock_pricesflex(simnum, sigma, mu, TimeToMaturity, Stockprice[0,i],T, N0)
        cashflow_matrix = np.zeros((simnum, (T * TimeToMaturity) + 1))
        exercisesteps = 50//N0
        for j in range(1, TimeToMaturity * T+1, 1):
            exercise = Stock_price_paths[:,j]<=Exerciseboundary[exercisesteps*(N0-TimeToMaturity)+j*exercisesteps]
            Stock_price_paths[exercise,j+1:]=0
            Stock_price_paths[~exercise, j] = 0
        #for j in range(M * T, -1, -1):
        for j in range(TimeToMaturity * T, 0, -1):
            Cashflow = Stock_price_paths[:,j]>0
            cashflow_matrix[Cashflow, j] = K - Stock_price_paths[Cashflow, j]
        #cashflow_disc = np.matmul(cashflow_matrix[:, 0:(M * T) + 1], np.transpose(disc_vector[0:M*T+1]))
        cashflow_disc = np.matmul(cashflow_matrix[:, 1:( TimeToMaturity * T) + 1], np.transpose(disc_vector[0:TimeToMaturity * T ]))
        LSM[i]  = np.sum(cashflow_disc) / simnum

    for i in range(0,N):
        M=N-i
        Stock_price_paths = sim_stock_pricesflex(simnum, sigma, mu, M, Stockprice[0,i]+epsilon,T, N0)
        cashflow_matrix = np.zeros((simnum, (T * M) + 1))
        exercisesteps = 50//N0
        for j in range(1, M * T+1, 1):
            exercise = Stock_price_paths[:,j]<=Exerciseboundary[exercisesteps*(N0-M)+j*exercisesteps]
            Stock_price_paths[exercise,j+1:]=0
            Stock_price_paths[~exercise, j] = 0
        #for j in range(M * T, -1, -1):
        for j in range(M * T, 0, -1):
            Cashflow = Stock_price_paths[:,j]>0
            cashflow_matrix[Cashflow, j] = K - Stock_price_paths[Cashflow, j]
        #cashflow_disc = np.matmul(cashflow_matrix[:, 0:(M * T) + 1], np.transpose(disc_vector[0:M*T+1]))
        cashflow_disc = np.matmul(cashflow_matrix[:, 1:(M * T) + 1], np.transpose(disc_vector[0:M * T ]))
        LSMepsilon[i]  = np.sum(cashflow_disc) / simnum
    Delta = (LSMepsilon-LSM)/epsilon
    return Delta, LSM, Stockprice

#NOT USED
def LSMdeltaresidual():
    Delta, LSM, Stockprice = sekantlsm(simnum, sigma, mu, N, T, S0, K, N0, epsilon,N2)
    Cash= np.zeros(N)
    RP = np.zeros(N+1)
    RP[0]=LSM[0]
    disc=np.exp(mu*(1/N0))
    Cash[0]=LSM[0]-Delta[0]*Stockprice[0,0]
    for i in range(1,N0):
        RP[i]=Cash[i-1]*disc+Delta[i-1]*Stockprice[0,i]
        Cash[i]=RP[i]-Delta[i]*Stockprice[0,i]
    RP[N] = Cash[N-1] * disc + Delta[N-1] * Stockprice[0, N]
    LSM=np.append(LSM,np.maximum(0,K-Stockprice[0,N]))
    return LSM, RP



# USED IN THE LSM HEDGING ERROR FUNKTION
#calculates the option price for each rebalacing point, by simulation simnum stock prices.
#hedgetime, number of delta, N0: initital number og rebalacing points
#N2: number of exercise points to estimate the boundary
def Price(Stockprice,Exerciseboundary,HedgeTime,N0,epsilon, simnum, sigma, mu,T,K,N2):
    disc_vector = d(mu, N0, T)
    Price = np.zeros(HedgeTime)
    exercisesteps = N2 // N0
    for i in range(0,HedgeTime):
            print(i)
            M=N0-i #M: number of points until maturity
            Eps=Stockprice[i]*epsilon #We let epsilon be a factor
            Stock_price_paths = sim_stock_pricesflex(simnum, sigma, mu, M, Stockprice[i]+Eps,T, N0)
            cashflow_matrix = np.zeros((simnum, (T * M) + 1))
            for j in range(1, M * T+1, 1):
                exercise = Stock_price_paths[:,j]<=Exerciseboundary[exercisesteps*i+j*exercisesteps]
                Stock_price_paths[exercise,j+1:]=0
                Stock_price_paths[~exercise, j] = 0
            #for j in range(M * T, -1, -1):
            for j in range(M * T, 0, -1):
                Cashflow = Stock_price_paths[:,j]>0
                cashflow_matrix[Cashflow, j] = K - Stock_price_paths[Cashflow, j]
            #cashflow_disc = np.matmul(cashflow_matrix[:, 0:(M * T) + 1], np.transpose(disc_vector[0:M*T+1]))
            cashflow_disc = np.matmul(cashflow_matrix[:, 1:(M * T) + 1], np.transpose(disc_vector[0:M * T ]))
            Price[i]  = np.sum(cashflow_disc) / simnum
    return Price

#USED IN THE SINGLE HEDGING ERROR FUNCTION
#function that finds the hedgeing time
def hedgetime(Stockprice,Exerciseboundary, N2,N0):
    exercisesteps = N2 // N0
    optionalive = np.zeros(N0 - 1)
    #AMoption_price, exerciseboundary = lsm(100000, sigma, mu, N2, T, S0, K)  # determines the boundary
    #Exerciseboundary = exerciseboundary * K
    #Stockprice = StockpricesDelta[0]  # from Brwonian_motion
    for i in range(1, N0):  # determines number of rebalacing points where the option is alive
        if Stockprice[i] > Exerciseboundary[i * exercisesteps]:
            optionalive[i - 1] = 1
        else:
            break
    HedgeTime = int(sum(optionalive)) + 1  # the time we need to delta hedge
    return HedgeTime #, Exerciseboundary, Stockprice

#USED IN THE LSM HEDGING ERROR FUNCTION
#determines the hedging time for each path in a stock price path matrix
def hedgetime2(StockpricesDelta,Exerciseboundary, N2,N0):
    exercisesteps = N2 // N0
    Hedgingtime= np.full(len(StockpricesDelta), np.nan)
    for j in range(len(StockpricesDelta)):
        stockprice=StockpricesDelta[j,:]
        optionalive = np.zeros(N0 - 1)
        for i in range(1, N0):  # determines number of rebalacing points where the option is alive
            if stockprice[i] > Exerciseboundary[i * exercisesteps]:
                optionalive[i - 1] = 1
            else:
                break
        Hedgingtime[j] = int(sum(optionalive)+1)   # the time we need to delta hedge
    return Hedgingtime

#THE LSM HEDGING ERROR FUNCTION
#N2:exercise points used to determine the boundary. #N0:number og repalancing points
def LSMsekant2( Exerciseboundary, StockpricesDelta, N0, epsilon, simnum, sigma, mu, T, K,N2):
    Hedgingtime=hedgetime2(StockpricesDelta,Exerciseboundary, N2,N0)
    Hedgingeror = np.full(len(StockpricesDelta), np.nan)

    for j in range(800,1000):
        print('stock path',j)
        Hedgingtimej= int(Hedgingtime[j])
        print('Hedgingtime', Hedgingtimej)
        StockpricesDeltaj=StockpricesDelta[j]
        LSM = Price(StockpricesDeltaj, Exerciseboundary,Hedgingtimej, N0, 0, 100000, sigma, mu, T, K, N2)
        LSMepsilon = Price(StockpricesDeltaj, Exerciseboundary, Hedgingtimej, N0, epsilon, 50000, sigma, mu, T, K, N2)
        Eps =  StockpricesDeltaj * epsilon
        Delta = (LSMepsilon - LSM) / Eps[:Hedgingtimej]
        Delta = np.clip(Delta, -1, 0)
        Cash = np.zeros(Hedgingtimej)
        RP = np.zeros(Hedgingtimej + 1)
        RP[0] = 4.4756271241618295
        disc = np.exp(mu * (1 / N0))
        Cash[0] = RP[0] - Delta[0] * StockpricesDeltaj[0]
        for i in range(1, Hedgingtimej):
            RP[i] = Cash[i - 1] * disc + Delta[i - 1] * StockpricesDeltaj[ i]
            Cash[i] = RP[i] - Delta[i] * StockpricesDeltaj[ i]
        RP[Hedgingtimej] = Cash[Hedgingtimej - 1] * disc + Delta[Hedgingtimej - 1] * StockpricesDeltaj[ Hedgingtimej]
        hedgerror = np.maximum(0, K - StockpricesDeltaj[Hedgingtimej]) - RP[Hedgingtimej]
        #LSM = np.append(LSM, np.maximum(0, K - StockpricesDelta[ Hedgingtime]))
        #hedgerror = RP[Hedgingtime] - LSM[Hedgingtime]
        timetomaturity = (N0 - Hedgingtimej) / N0
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        Hedgingeror[j]=dischedgerror
        # return Delta, LSM, Stockprice, dischedgerror, RP
    return Hedgingeror

def LSMsekant2( Exerciseboundary, StockpricesDelta, N0, epsilon, simnum, sigma, mu, T, K,N2):
    Hedgingtime=hedgetime2(StockpricesDelta,Exerciseboundary, N2,N0)
    HedgingerorLSM100000 = np.full(len(StockpricesDelta), np.nan)

    for j in range(100,200):
        print('stock path',j)
        Hedgingtimej= int(Hedgingtime[j])
        print('Hedgingtime', Hedgingtimej)
        StockpricesDeltaj=StockpricesDelta[j]
        LSM = Price(StockpricesDeltaj, Exerciseboundary,Hedgingtimej, N0, 0, 100000, sigma, mu, T, K, N2)
        LSMepsilon = Price(StockpricesDeltaj, Exerciseboundary, Hedgingtimej, N0, epsilon, 100000, sigma, mu, T, K, N2)
        Eps =  StockpricesDeltaj * epsilon
        Delta = (LSMepsilon - LSM) / Eps[:Hedgingtimej]
        Delta = np.clip(Delta, -1, 0)
        Cash = np.zeros(Hedgingtimej)
        RP = np.zeros(Hedgingtimej + 1)
        RP[0] = 4.4756271241618295
        disc = np.exp(mu * (1 / N0))
        Cash[0] = RP[0] - Delta[0] * StockpricesDeltaj[0]
        for i in range(1, Hedgingtimej):
            RP[i] = Cash[i - 1] * disc + Delta[i - 1] * StockpricesDeltaj[ i]
            Cash[i] = RP[i] - Delta[i] * StockpricesDeltaj[ i]
        RP[Hedgingtimej] = Cash[Hedgingtimej - 1] * disc + Delta[Hedgingtimej - 1] * StockpricesDeltaj[ Hedgingtimej]
        hedgerror = np.maximum(0, K - StockpricesDeltaj[Hedgingtimej]) - RP[Hedgingtimej]
        #LSM = np.append(LSM, np.maximum(0, K - StockpricesDelta[ Hedgingtime]))
        #hedgerror = RP[Hedgingtime] - LSM[Hedgingtime]
        timetomaturity = (N0 - Hedgingtimej) / N0
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        HedgingerorLSM100000[j]=dischedgerror
        # return Delta, LSM, Stockprice, dischedgerror, RP
    return HedgingerorLSM100000

def plotdeltadistLSM(Exerciseboundary, StockpricesDelta, N0, epsilon, simnum, sigma, mu, T, K,N2):
    Hedgingeror=LSMsekant2( Exerciseboundary, StockpricesDelta, N0, epsilon, simnum, sigma, mu, T, K,N2)
    plt.subplots(figsize=(10, 6))
    sns.set(style="darkgrid")
    colors = sns.color_palette("Purples_d", 6)
    plt.hist( Hedgingerorbackup, bins=30, label='50 ', color=colors[2],density=True)
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.title('Hedging error distribution - LSM')
    #plt.legend(['50'], loc='upper left', bbox_to_anchor=(0, 1), title="rebalancing points")
    plt.show()

#USED IN THE SINGLE HEDGING ERROR FUNCTION
#function that finds the hedgeing time
def hedgetime(Stockprice,Exerciseboundary, N2,N0,sigma, mu, T, S0, K):
    exercisesteps = N2 // N0
    optionalive = np.zeros(N0 - 1)
    #AMoption_price, exerciseboundary = lsm(100000, sigma, mu, N2, T, S0, K)  # determines the boundary
    #Exerciseboundary = exerciseboundary * K
    #Stockprice = StockpricesDelta[0]  # from Brwonian_motion
    for i in range(1, N0):  # determines number of rebalacing points where the option is alive
        if Stockprice[i] > Exerciseboundary[i * exercisesteps]:
            optionalive[i - 1] = 1
        else:
            break
    HedgeTime = int(sum(optionalive)) + 1  # the time we need to delta hedge
    return HedgeTime #, Exerciseboundary, Stockprice

#SINGLE HEDGING ERROR FUNKTION
#can be usen for one stock price
def LSMsekant(HedgeTime, Exerciseboundary, Stockprice, N0, epsilon, simnum, sigma, mu, T, K):
    LSM = Price(Stockprice,Exerciseboundary,HedgeTime,N0,0, simnum, sigma, mu,T,K,N2)
    LSMepsilon = Price(Stockprice,Exerciseboundary,HedgeTime,N0,epsilon, simnum, sigma, mu,T,K,N2)
    Eps = np.zeros(HedgeTime)
    for i in range(0, HedgeTime):
        Eps[i] = Stockprice[0, i] * epsilon
    Delta = (LSMepsilon - LSM) / Eps
    Cash = np.zeros(HedgeTime)
    RP = np.zeros(HedgeTime + 1)
    RP[0] = LSM[0]
    disc = np.exp(mu * (1 / N0))
    Cash[0] = LSM[0] - Delta[0] * Stockprice[0, 0]
    for i in range(1, HedgeTime):
        RP[i] = Cash[i - 1] * disc + Delta[i - 1] * Stockprice[0, i]
        Cash[i] = RP[i] - Delta[i] * Stockprice[0, i]
    RP[HedgeTime] = Cash[HedgeTime - 1] * disc + Delta[HedgeTime - 1] * Stockprice[0, HedgeTime]
    LSM = np.append(LSM, np.maximum(0, K - Stockprice[0, HedgeTime]))
    hedgerror = RP[HedgeTime] - LSM[HedgeTime]
    timetomaturity = (N0 - HedgeTime) / N0
    dischedgerror = hedgerror * np.exp(mu * timetomaturity)
    #return Delta, LSM, Stockprice, dischedgerror, RP
    return dischedgerror


#NOT USED
#Distribution with the LSM sekant method
    deltaerror=np.zeros(100)
    for i in range(100):
        HedgeTime, Exerciseboundary, Stockprice =  hedgetime(exerciseboundary, N2,N0,sigma, mu, T, S0, K)
        deltaerror[i]=LSMsekant(HedgeTime, Exerciseboundary, Stockprice, N0, epsilon, simnum, sigma, mu, T, K)
        print(i)
    return deltaerror

#NOT USED
def LSMdeltaresidual():
    Delta, LSM, Stockprice = sekantlsm(simnum, sigma, mu, N, T, S0, K, N0, epsilon)
    Cash= np.zeros(N)
    RP = np.zeros(N+1)
    RP[0]=LSM[0]
    disc=np.exp(mu*(1/N0))
    Cash[0]=LSM[0]-Delta[0]*Stockprice[0,0]
    for i in range(1,N0):
        RP[i]=Cash[i-1]*disc+Delta[i-1]*Stockprice[0,i]
        Cash[i]=RP[i]-Delta[i]*Stockprice[0,i]
    RP[N] = Cash[N-1] * disc + Delta[N-1] * Stockprice[0, N]
    LSM=np.append(LSM,np.maximum(0,K-Stockprice[0,N]))
    return LSM, RP

#Stockprice=np.array([36., 36.61639618, 38.30728035, 36.37648554, 36.66202668,
       # 34.881435, 33.91316083, 32.46307103, 35.84760912, 34.86874525,
       # 34.80294035])
#X = np.arange(0, T + 0.1, 0.1)
#X2=np.arange(0, T + 0.02, 0.02)
#plt.plot(X, Stockprice.T)
#plt.plot(X2, Exerciseboundary)
#plt.show()

LSMprice1
LSM1column_averages = np.mean(LSMprice1[0:50], axis=0)
LSM1column_std_dev = np.std(LSMprice1[0:50], axis=0, ddof=1)

LSMprice2
LSM2column_averages = np.mean(LSMprice2[0:50], axis=0)
LSM2column_std_dev = np.std(LSMprice2[0:50], axis=0, ddof=1)

LSMprice3
LSM3column_averages = np.mean(LSMprice3[0:50], axis=0)
LSM3column_std_dev = np.std(LSMprice3[0:50], axis=0, ddof=1)

LSMprice4
LSM3column_averages = np.mean(LSMprice4[0:50], axis=0)
LSM3column_std_dev = np.std(LSMprice4[0:50], axis=0, ddof=1)

LSMprice5,LSMprice6, LSMprice7, LSMprice8

LSMprice9
LSM9column_averages = np.mean(LSMprice9[0:50], axis=0)
LSM9column_std_dev = np.std(LSMprice9[0:50], axis=0, ddof=1)

LSMprice10
LSM10column_averages = np.mean(LSMprice10[0:50], axis=0)
LSM10column_std_dev = np.std(LSMprice10[0:50], axis=0, ddof=1)

LSMprice11
LSM11column_averages = np.mean(LSMprice11[0:50], axis=0)
LSM11column_std_dev = np.std(LSMprice11[0:50], axis=0, ddof=1)

LSMprice12
LSM12column_averages = np.mean(LSMprice12[0:50], axis=0)
LSM12column_std_dev = np.std(LSMprice12[0:50], axis=0, ddof=1)

LSMprice13, LSMprice14, LSMprice15, LSMprice16

LSMprice17
LSM17column_averages = np.mean(LSMprice17[0:50], axis=0)
LSM17column_std_dev = np.std(LSMprice17[0:50], axis=0, ddof=1)

LSMprice18
LSM18column_averages = np.mean(LSMprice18[0:50], axis=0)
LSM18column_std_dev = np.std(LSMprice18[0:50], axis=0, ddof=1)

LSMprice19
LSM19column_averages = np.mean(LSMprice19[0:50], axis=0)
LSM19column_std_dev = np.std(LSMprice19[0:50], axis=0, ddof=1)

LSMprice20
LSM19column_averages = np.mean(LSMprice20[0:50], axis=0)
LSM19column_std_dev = np.std(LSMprice20[0:50], axis=0, ddof=1)

t1_start = process_time()
AMoption_price, exerciseboundary = lsm(100000, 0.2, 0.06, 50, 1, 36, 40)
t1_stop = process_time()
t1_stop-t1_start
