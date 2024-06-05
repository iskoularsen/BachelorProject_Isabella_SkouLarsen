import numpy as np
from Brownian_motion import *
import matplotlib.pyplot as plt
from time import process_time

def latticebinom(S0,T,N,sigma):
    dt = 1 / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    Stopckprices = np.full((N+1,N+1), np.nan)
    for i in np.arange(T*N,-1,-1):
        Stopckprices[0:i+1, i] = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
    return Stopckprices

def latticebinomplot(S0,T,N,sigma):
    dt=1/N
    Stopckprices=latticebinom(S0,T,N,sigma)
    fig, ax = plt.subplots(figsize=(12, 8))
    for i in range(N + 1):
        for j in range(i + 1):
            # Line to down
            if i + 1 < N + 1:
                # Line to down
                ax.plot([i, i + 1], [Stopckprices[j, i], Stopckprices[j, i + 1]], 'ko:', lw=1, markersize=2)
            if j + 1 < i + 2 and i + 1 < N + 1:
                # Line to up
                ax.plot([i, i + 1], [Stopckprices[j, i], Stopckprices[j + 1, i + 1]], 'ko:', lw=1, markersize=2)
            # Annotate the node with the stock price
            ax.text(i, Stopckprices[j, i] + 0.2, f'{Stopckprices[j, i]:.2f}', ha='center', va='bottom', fontsize=8)
    # Set labels and grid
    ax.set_title('Binomial Tree for Stock Prices')
    ax.set_xlabel('Time')
    ax.set_ylabel('Stock price')
    ax.set_xticks(range(N + 1))

    # Set x-tick labels to correspond to the actual time increments
    ax.set_xticklabels([f'{i * dt:.1f}' for i in range(N + 1)])
    plt.grid(True)
    #ax.set_xticks(np.arange(0, N + dt, dt))
    plt.show()


#Can be used to determine th optionprice, Time 0 price, and the exercise boundary
#K:Strike price, S0:initial stock price, T:maturity time in years, N:number of excersise points,
# r:risk free rate, sigma:volatility
def AMbinom_price(K,S0,T,N,r,sigma):
    dt = 1 / N
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    q = (np.exp(r*dt) - d) / (u - d)
    disc = np.exp(-r*dt)
    S = S0 * d ** (np.arange(T*N, -1, -1)) * u ** (np.arange(0, T*N + 1, 1))
    C = np.maximum(0, K - S)
    #Exerciseboundary=np.zeros(int(T/dt),)
    Exerciseboundary = np.zeros(N*T)

    for i in np.arange(T*N-1,0,-1):
        S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
        C[:i + 1] = disc * (q * C[1:i + 2] + (1 - q) * C[0:i + 1])
        C = C[:-1]
        dummy = K - S >= C
        delta=(C[1]-C[0])/(S[1]-S[0])
        if np.any(dummy):
            last_true_index = np.where(dummy)[0][-1]
            Exerciseboundary[i-1] = S[last_true_index]
        else:
            Exerciseboundary[i-1] = np.nan
        C = np.maximum(C, K - S)
    S = S0 * d ** (np.arange(0, -1, -1)) * u ** (np.arange(0,  1, 1))
    C[: 1] = disc * (q * C[1: 2] + (1 - q) * C[0: 1])
    C = C[:-1]
    dummy = K - S >= C
    if np.any(dummy):
        last_true_index = np.where(dummy)[0][-1]
        Exerciseboundary[ - 1] = S[last_true_index]
    else:
        Exerciseboundary[ - 1] = np.nan
    C = np.maximum(C, K - S)
    Exerciseboundary[T * N - 1] = K
    return C , delta, Exerciseboundary


AMbinom_price(36,40,1,2500,0.06,0.2)
def binomboudaryplot(K, S0, T, r, sigma):
    C, delta, Exerciseboundary50 = AMbinom_price(K, S0, T, 50, r, sigma)
    C, delta, Exerciseboundary1000 = AMbinom_price(K, S0, T, 1000, r, sigma)
    C, delta, Exerciseboundary10000 = AMbinom_price(K, S0, T, 10000, r, sigma)
    dt1 = 1 / 50
    dt2 = 1 / 1000
    dt4 = 1 / 10000
    X2 = np.arange(dt2, T +dt2, dt2)
    X= np.arange(dt1, T + dt1, dt1)
    X4 = np.arange(dt4, T + dt4, dt4)
    plt.subplots(figsize=(10, 6))
    sns.set(style="darkgrid")
    colors = sns.color_palette("GnBu", 10)
    plt.plot(X, Exerciseboundary50, label='Boundary 50 steps',color=colors[5] )
    plt.plot(X2, Exerciseboundary1000, label='Boundary 1000 steps', color=colors[7], alpha=0.8)
    plt.plot(X4, Exerciseboundary10000, label='Boundary 10000 steps',color=colors[9])
    plt.xlabel('Time')
    plt.ylabel('Stock price')
    plt.title('Exercise boundary - The Binomial model')
    legend1 = plt.legend(['50', '1,000', '10,000'], loc='upper left', bbox_to_anchor=(0, 1), title="Exercise points")
    # Add the first legend manually
    plt.gca().add_artist(legend1)
    # Create the second legend, placed next to the first
    legend2 = plt.legend(['0.0022 s', '0.0325 s', '1.0485 s'], loc='upper left', bbox_to_anchor=(0.2, 1),
                         title="ART")
    plt.show()


  t1_start = process_time()
    AMbinom_price(K,S0,T,2500,r,sigma)
    t1_stop = process_time()

def tabel3():
    optionpricebinom = np.full(502, np.nan)
    for i in range(2, 502,1):
        print(i)
        AMoption_price, delta, exerciseboundary = AMbinom_price(40,36,1,i,0.06,0.2)
        optionpricebinom[i]= AMoption_price
def convergencebinomplot():
    optionprice = np.full(2500, np.nan)
    error = np.full(2500, np.nan)
    for i in range(2, 2500):
        optionprice[i] = AMbinom_price(K, S0, T, i, r, sigma)[0]
    for i in range(2, 2499):
        error[i] = (optionprice[i] - optionprice[i + 1])/optionprice[i + 1]*100
    plt.subplots(figsize=(10, 6))
    sns.set(style="darkgrid")
    colors = sns.color_palette("GnBu", 10)
    #plt.plot(optionprice, linewidth=1, color=colors[7])
    plt.plot(np.abs(error), linewidth=1, color=colors[7])
    #convergence_value = optionprice[~np.isnan(optionprice)][-1]
   # plt.axhline(y=convergence_value, color='red', alpha=0.5, linestyle=':', linewidth=1,
    #           label=f'Convergence Value: {convergence_value:.4f}')
    plt.xlabel('Time steps')
    plt.ylabel('Percentage deviation')
    plt.title('Percentage deviation of Binomial model prices')
    plt.show()





error = np.full(2500, np.nan)
for i in range(2,2199):
    error[i]=optionprice[i]-optionprice[i+1]

indices = np.arange(2, 2200, 5)# From 2 to 2499, every 20th index

# Extract the option prices for these indices
prices_to_plot = optionprice[indices]

# Plotting
plt.figure(figsize=(12, 6))
plt.plot(indices, prices_to_plot, 'bo-', label='Option Price')
plt.xlabel('Number of Steps')
plt.ylabel('American Option Price')
plt.title('American Option Price vs. Number of Steps (Every 20th Step)')
plt.grid(True)
plt.legend()
plt.show()


N = 50
S0  = 36
T = 1
sigma = 0.2
K =40
r = 0.06

price, delta, Stockprice =AMbinom_price(K,S0,T,N,r,sigma)
D=Stockprice
plt.plot(D)
plt.show()

#Used to plot how delta is effected by the stockprice
def Delta(S0, K, T,N,r,sigma):
    Stockprices = S0 + np.arange(-S0, S0, 1)
    Delta =  np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        C, delta ,Exerciseboundary = AMbinom_price(K, Stockprices[i-1], T, N, r, sigma)
        Delta[i-1]=delta
    return Delta

def plotdeltabinom():
    Deltabinom = Delta(S0, K, 1, N, r, sigma)
    plt.figure(figsize=(10, 4))
    sns.set(style="darkgrid")
    X4 = np.arange(16, 76, 1)
    colors = sns.color_palette("GnBu", 10)
    plt.plot(Delta1, 'o-',markersize=3,label='J=2.500 ', color=colors[7])
    plt.plot(X4,Delta4, 'o-',markersize=3,label='J=50.000', color=colors[3])
    plt.xlim((16,70))
    plt.xlabel('Option Price')
    plt.ylabel('Delta')
    plt.title('Delta function - Binomial model')
    plt.legend()
    plt.show()


Delta1=Delta(36, 40, 1,2500,0.06,0.2)
Delta2=Delta(S0, K, 2,N,r,sigma)
Delta3=Delta(S0, K, 1,1000,r,sigma)
Delta4=Delta(S0, K, 1,50000,r,sigma)
#This is used for the Delta experiment to determine the exercise boundary
def AMbinom_price2(K,S0,T,N,r,sigma,dt):
    Totalsteps=N*T
    u = np.exp(sigma * np.sqrt(dt))
    d = np.exp(-sigma * np.sqrt(dt))
    q = (np.exp(r*dt) - d) / (u - d)
    disc = np.exp(-r*dt)
    S = S0 * d ** (np.arange(Totalsteps, -1, -1)) * u ** (np.arange(0, Totalsteps + 1, 1))
    C = np.maximum(0, K - S)
    Stockprice=np.zeros(int(T/dt),)
    Stockprice[T*N-1]=K


    for i in np.arange(Totalsteps-1,0,-1):
        S = S0 * d ** (np.arange(i, -1, -1)) * u ** (np.arange(0, i + 1, 1))
        C[:i + 1] = disc * (q * C[1:i + 2] + (1 - q) * C[0:i + 1])
        C = C[:-1]
        dummy = K - S >= C
        delta=(C[1]-C[0])/(S[1]-S[0])
        if np.any(dummy):
            last_true_index = np.where(dummy)[0][-1]
            Stockprice[i-1] = S[last_true_index]
        else:
            Stockprice[i-1] = np.nan
        C = np.maximum(C, K - S)
    delta = (C[1] - C[0]) / (S[1] - S[0])
    S = S0 * d ** (np.arange(0, -1, -1)) * u ** (np.arange(0,  1, 1))
    C[: 1] = disc * (q * C[1: 2] + (1 - q) * C[0: 1])
    C = C[:-1]
    dummy = K - S >= C
    if np.any(dummy):
        last_true_index = np.where(dummy)[0][-1]
        Stockprice[ - 1] = S[last_true_index]
    else:
        Stockprice[ - 1] = np.nan
    C = np.maximum(C, K - S)
    return C , delta, Stockprice

def plotdeltadistbinom():
    Delta1binom=BinomDeltaeksp(0.2, 0.06, 50, 1, 36, 10000, StockpricesDelta, 40)
    Delta2binom=BinomDeltaeksp(sigma, mu, 25, T, S0, 10000, StockpricesDelta2, K)
    Delta3binom=BinomDeltaeksp(sigma, mu, 10, T, S0, 10000, StockpricesDelta1, K)
    plt.subplots(figsize=(12, 8))
    sns.set(style="darkgrid")
    colors = sns.color_palette("GnBu", 10)
    plt.hist( Delta3binom, bins=30, label='10 rebalancing points', color=colors[5])
    plt.hist( Delta2binom, bins=30,  label='25 rebalancing points', color=colors[7])
    plt.hist( Delta1binom, bins=30,label='50 rebalancing points', color=colors[9])
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Hegding error distribution - Binomial model')
    plt.legend(['10', '25', '50'], loc='upper left', bbox_to_anchor=(0, 1), title="rebalancing points")
    plt.show()

StockpricesDelta=sim_stock_prices2(1000, 0.2, 0.06, 50, 1, 36)
StockpricesDelta1=sim_stock_prices2(1000, sigma, mu, 10, T, S0)
StockpricesDelta2=sim_stock_prices2(1000, sigma, mu, 25, T, S0)

#calculates the delta given a stockprice
#sigma:volatility, mu:risk free rate, N:number of rebalacing points, T:Maturity in years, S0: initial stockprice
#N2: number of excercise points the boundary should be estimated from
def BinomDeltaeksp(sigma, mu, N, T, S0,N2,StockpricesDelta,K):
    start = time.time()
    dt=1/N
    exercisesteps = N2 // N
    dt2=1/N2
    HedgingerrorD2=np.zeros(len(StockpricesDelta))
    C, delta, Exerciseboundarybinom = AMbinom_price2(K,S0,T,N2,mu,sigma,dt2)
    for j in range(0,len(StockpricesDelta)):
        optionalive = np.zeros(N - 1)
        stock_price_path=StockpricesDelta[j,:]
        for i in range(1,N):
            if stock_price_path[i]>Exerciseboundary[i*exercisesteps]:
                optionalive[i - 1] = 1
            else:
                break
        HedgeTime = int(sum(optionalive)) + 1
        Delta = np.zeros(HedgeTime)
        Optionprice = np.zeros(HedgeTime+1)
        #calculating the delta for each point in the stock price path, by making a new binomial tree.
        for i in range(HedgeTime):
            C, delta, Stockprice = AMbinom_price2(K, stock_price_path[i], T, N-i, mu, sigma,dt)
            Delta[i] = delta
            Optionprice[i] = C[0]
        #Cashflow obtained from exercing optimally
        Optionprice[HedgeTime]=np.maximum(0,K-stock_price_path[HedgeTime])
        #return dischedgerror
        Cash = np.zeros(HedgeTime)
        RP = np.zeros(HedgeTime + 1)
        RP[0] = Optionprice[0]
        disc = np.exp(mu * (1 / N))
        Cash[0] = Optionprice[0] - Delta[0] * stock_price_path[ 0]
        for i in range(1, HedgeTime):
            RP[i] = Cash[i - 1] * disc + Delta[i - 1] * stock_price_path[ i]
            Cash[i] = RP[i] - Delta[i] * stock_price_path[ i]
        RP[HedgeTime] = Cash[HedgeTime - 1] * disc + Delta[HedgeTime - 1] * stock_price_path[HedgeTime]
        hedgerror = Optionprice[HedgeTime]-RP[HedgeTime]
        timetomaturity = (N - HedgeTime) / N
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        HedgingerrorD2[j]=dischedgerror
        end = time.time()
        print(end - start)
    return Hedgingerror




#Is used to fin the distribution of the errors
#sigma:volatility, mu:risk free rate, N:number of rebalacing points, T:Maturity in years, S0: initial stockprice
#N2: number of excercise points the boundary should be estimated from, M:number of errors to find
#DOES NOT WORK ANYMORE
def distdeltabinom(sigma, mu, N, T, S0,N2,M):
    deltaerror=np.zeros(M)
    for i in range(M):
        deltaerror[i]=BinomDeltaeksp(sigma, mu, N, T, S0,N2)
    return deltaerror


#calculates the hedging error discountet to time T
def deltaerror(HedgeTime, Optionprice, mu, N, Delta,stock_price_path):
    Cash = np.zeros(HedgeTime)
    RP = np.zeros(HedgeTime + 1)
    RP[0] = Optionprice[0]
    disc = np.exp(mu * (1 / N))
    Cash[0] = Optionprice[0] - Delta[0] * stock_price_path[0]
    for i in range(1, HedgeTime):
        RP[i] = Cash[i - 1] * disc + Delta[i - 1] * stock_price_path[i]
        Cash[i] = RP[i] - Delta[i] * stock_price_path[i]
    RP[HedgeTime] = Cash[HedgeTime - 1] * disc + Delta[HedgeTime - 1] * stock_price_path[HedgeTime]
    hedgerror = RP[HedgeTime] - Optionprice[HedgeTime]
    timetomaturity = (N - HedgeTime) / N
    dischedgerror = hedgerror * np.exp(mu * timetomaturity)
    return dischedgerror


stock_price_path= np.array([36., 36.48642014, 37.86151866, 34.8456401, 34.73871265,
            34.12517739, 35.4861796, 33.37300352, 34.69088191, 36.04537846,
            33.22865797])
N = 10
S0  = 36
T = 1
sigma = 0.2
K =40
r = 0.06
N2=2000



#https://quantpy.com.au/binomial-tree-model/american-put-options-with-the-binomial-asset-pricing-model/