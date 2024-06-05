import matplotlib.pyplot as plt
import scipy.stats as si
from Brownian_motion import * #used to simulate stock price path
from regression import * #uses the LSM for exerciseboundary and option price

simnum = 100000
sigma = 0.2
mu=0.06
N = 10
T = 1
S0 = 36
K = 40
N2=50

#Determines the exercise boundary and time zero option price
def exerciseboundary(simnum, sigma, mu, N2, T, S0, K):
    AMoption_price, exerciseboundary = lsm(simnum, sigma, mu, N2, T, S0, K)
    Exerciseboundary = exerciseboundary * K
    return AMoption_price, Exerciseboundary

AMoption_price, Exerciseboundary = exerciseboundary(simnum, sigma, mu, 50, T, S0, K)
Stockpricestest=sim_stock_prices2(1000, 0.2, 0.06, 50, 1, 36)
#uses the same exercise boundary and time zero option price. Is used to determine the delta using a Balck-Scholes model
def DeltaBlackSchoels(T,N,sigma,mu,K, Exerciseboundary,StockpricesDelta,N2):
    dt=1/N
    exercisesteps = N2 // N
    Hedgingerror = np.zeros(len(StockpricesDelta))
    for j in range(0, len(StockpricesDelta)):
        Stockprice = StockpricesDelta[j, :]
        optionalive = np.zeros(N - 1)
        for i in range(1, N):
            if Stockprice[i] > Exerciseboundary[i * exercisesteps]:
                optionalive[i - 1] = 1
            else:
                break
        HedgeTime = int(sum(optionalive))+1
        Delta = np.zeros(HedgeTime)
        for i in range(HedgeTime):
            Delta[i]=Deltaput(Stockprice[i],K,T,mu,sigma,i*dt)
        Cash = np.zeros(HedgeTime)
        RP = np.zeros(HedgeTime + 1)
        RP[0] = 4.4756271241618295
        disc = np.exp(mu * (1 / N))
        Cash[0] = RP[0] - Delta[0] * Stockprice[ 0]
        for i in range(1, HedgeTime):
            RP[i] = Cash[i - 1] * disc + Delta[i - 1] * Stockprice[i]
            Cash[i] = RP[i] - Delta[i] * Stockprice[i]
        RP[HedgeTime] = Cash[HedgeTime - 1] * disc + Delta[HedgeTime - 1] * Stockprice[HedgeTime]
        hedgerror =  np.maximum(0, K - Stockprice[HedgeTime])-RP[HedgeTime]
        timetomaturity = (N - HedgeTime) / N
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        Hedgingerror[j] = dischedgerror
    return Hedgingerror

def plotdeltadistbinom(StockpricesDelta,Exerciseboundary, T,N,sigma,mu,K):
    Hedgingerror=DeltaBlackSchoels(T,N,sigma,mu,K, Exerciseboundary,StockpricesDelta,N2)
    plt.subplots(figsize=(10, 6))
    sns.set(style="darkgrid")
    colors = sns.color_palette("gist_yarg", 6)
    plt.hist( Hedgingerror, bins=30, color=colors[3], density=True)
    plt.xlabel('Error')
    plt.ylabel('Density')
    plt.title('Hedging error distribution - Black-Scholes model')
    plt.show()

def plotdeltadistbinom2():
    Hedgingerror1, HedgingerrorT=DeltaBlackSchoels2(T,N,sigma,mu,K, Exerciseboundary,Stockpricestest,N2)
    plt.subplots(figsize=(10, 6))
    sns.set(style="darkgrid")
    colors = sns.color_palette("gist_yarg", 6)
    plt.hist(Hedgingerror1, bins=20, alpha=0.75, label='Hedging Error',  color=colors[5])
    plt.hist(HedgingerrorT, bins=20, alpha=0.75, label='Hedging Error T', color=colors[1])
    plt.xlabel('Error')
    plt.ylabel('Frequency')
    plt.title('Hedging error Frequency - Black-Scholes model')
    plt.legend(['Early exercise', 'Exercise at maturity'])
    plt.show()

Stockpricestest=sim_stock_prices2(10000, 0.2, 0.06, 50, 1, 36)
def DeltaBlackSchoels2(T,N,sigma,mu,K, Exerciseboundary,StockpricesDelta,N2):
    dt=1/N
    exercisesteps = N2 // N
    HedgingerrorT2 = np.full(len(StockpricesDelta), np.nan)
    Hedgingerrorearly = np.full(len(StockpricesDelta), np.nan)
    for j in range(0, len(StockpricesDelta)):
        Stockprice = StockpricesDelta[j, :]
        optionalive = np.zeros(N - 1)
        for i in range(1, N):
            if Stockprice[i] > Exerciseboundary[i * exercisesteps]:
                optionalive[i - 1] = 1
            else:
                break
        HedgeTime = int(sum(optionalive))+1
        Delta = np.zeros(HedgeTime)
        for i in range(HedgeTime):
            Delta[i]=Deltaput(Stockprice[i],K,T,mu,sigma,i*dt)
        Cash = np.zeros(HedgeTime)
        RP = np.zeros(HedgeTime + 1)
        RP[0] = 4.4756271241618295
        disc = np.exp(mu * (1 / N))
        Cash[0] = RP[0] - Delta[0] * Stockprice[ 0]
        for i in range(1, HedgeTime):
            RP[i] = Cash[i - 1] * disc + Delta[i - 1] * Stockprice[i]
            Cash[i] = RP[i] - Delta[i] * Stockprice[i]
        RP[HedgeTime] = Cash[HedgeTime - 1] * disc + Delta[HedgeTime - 1] * Stockprice[HedgeTime]
        hedgerror =  np.maximum(0, K - Stockprice[HedgeTime])-RP[HedgeTime]
        timetomaturity = (N - HedgeTime) / N
        dischedgerror = hedgerror * np.exp(mu * timetomaturity)
        if HedgeTime==50:
            HedgingerrorT2[j] = dischedgerror
        else:
            Hedgingerrorearly[j] = dischedgerror
    return Hedgingerror, HedgingerrorT


#calculates the delta distribution
def distdeltaBS(T,N,sigma,mu,S0,K,simnum, N2,M):
    deltaerror=np.zeros(M)
    for i in range(M):
        deltaerror[i]=DeltaBlackSchoels(T,N,sigma,mu,S0,K,AMoption_price, Exerciseboundary)
    return deltaerror

#calculates the delta of a put option using the Balck-Scholes model
def Deltaput(S0,K,T,mu,sigma,t):
    d1 = (np.log(S0 / K) + (mu + (sigma ** 2) / 2) * (T-t)) / (sigma * np.sqrt(T-t))
    Delta= -1*si.norm.cdf(-d1, 0,1)
    return Delta

def DeltaBS(S0,K,T,mu,sigma,t):
    Stockprices = S0 + np.arange(-S0, S0, 1)
    DeltaBS =  np.zeros(len(Stockprices))
    for i in range(len(Stockprices)):
        delta =Deltaput(Stockprices[i-1],K,T,mu,sigma,t)
        Delta[i-1]=delta
    return DeltaBS

def plotdeltabinom(S0,K,T,mu,sigma,t):
    Delta1 = DeltaBS(S0,K,T,mu,sigma,t)
    plt.figure(figsize=(10, 6))
    sns.set(style="darkgrid")
    colors1 = sns.color_palette("gist_yarg", 6)
    colors = sns.color_palette("GnBu", 10)
    plt.plot(X, Deltabinom[36 - 20:36 + 36], 'o-',markersize=3, label='BM', color=colors[7])
    plt.plot(Delta1, 'o-',markersize=3,label='Delta', color=colors1[5])
    plt.xlim((16,70))
    plt.xlabel('Option Price')
    plt.ylabel('Delta')
    plt.title('Delta function - Black-Scholes model')
    plt.show()

#not used but can determine the option price for put and call option using the Black-Schoels model
def BlackScholes(S0,K,T,mu,sigma):
    d1=(np.log(S0/K)+(mu+(sigma**2)/2)*T)/(sigma*np.sqrt(T))
    d2=d1-sigma*np.sqrt(T)
    #for put option
    P=K*np.exp(-mu*T)*si.norm.cdf(-d2,0,1)-S0*si.norm.cdf(-d1, 0,1)
    #for call option
    #C=-K*np.exp(-r*T)*si.norm.cdf(d2,0,1)+S0*si.norm.cdf(d1, 0,1)
    return P, d1,d2
