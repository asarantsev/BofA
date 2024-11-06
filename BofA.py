import pandas
import numpy 
from matplotlib import pyplot as plt
from scipy import stats
from statsmodels.tsa.stattools import adfuller
from statsmodels.api import OLS

ratings = ['Quality', 'Junk']
commonDF = pandas.read_excel('ratings.xlsx', sheet_name = 'common')
vix = commonDF['Volatility'].values
bill = commonDF['Risk-free'].values
N = len(bill)
keys = ['Yield', 'Spread', 'Excess']
file = open('results.txt', 'w')
allResid = [] # all residuals

def analysis(residuals):
    file.write('Original Residuals\n')
    file.write('Skewness = ' + str(stats.skew(residuals)) + ' Kurtosis = ' + str(stats.kurtosis(residuals)) + '\n\n')
    nresiduals = residuals/vix[1:]
    file.write('Normalized Residuals\n')
    file.write('Skewness = ' + str(stats.skew(nresiduals)) + ' Kurtosis = ' + str(stats.kurtosis(nresiduals)) + '\n\n')
    
for rating in ratings:
    file.write(rating + '\n')
    ratingDF = pandas.read_excel('ratings.xlsx', sheet_name = rating)
    returns = numpy.diff(numpy.log(ratingDF['Wealth']))
    premia = returns - numpy.log(1 + commonDF['Risk-free'].iloc[:-1]/1200)
    ratingDF.insert(3, 'Excess', ratingDF['Yield'] - commonDF['Risk-free'], True)
    for key in keys:
        file.write('AR for ' + key + ' and rating ' + rating + '\n\n')
        series = ratingDF[key]
        file.write('ADF test = ' + str(adfuller(series, maxlag = 15)) + '\n')
        dseries = numpy.diff(series)
        Reg = stats.linregress(series[:-1], dseries)
        s = Reg.slope
        i = Reg.intercept
        residuals = numpy.array([dseries[k] - s * series[k] - i for k in range(N-1)])
        file.write('slope = ' + str(s) + ' intercept = ' + str(i) + ' p = ' + str(Reg.pvalue) + '\n\n')
        analysis(residuals)
        RegDF = pandas.DataFrame({'const' : 1/vix[1:], 'lag' : series[:-1]/vix[1:], 'vix' : 1})
        Reg = OLS(dseries/vix[1:], RegDF).fit()
        allResid.append(Reg.resid)
        file.write('\n\n\n')
        file.write(str(Reg.summary()))
        file.write('\n\n\n')
    file.write('Duration regression of returns minus normalized rate upon rate change\n\n\n')
    series = ratingDF['Yield'].values/100
    dseries = numpy.diff(series)
    nreturns = returns - series[:-1]/12
    Reg = stats.linregress(dseries, nreturns)
    s = Reg.slope
    i = Reg.intercept
    residuals = numpy.array([nreturns[k] - s * dseries[k] - i for k in range(N-1)])
    file.write(str(Reg) + '\n\n')
    analysis(residuals)
    file.write('Same regression after normalization\n\n\n')
    RegDF = pandas.DataFrame({'const' : 1/vix[1:], 'duration' : dseries/vix[1:], 'vix' : 1})
    Reg = OLS(nreturns/vix[1:], RegDF).fit()
    allResid.append(Reg.resid)
    file.write(str(Reg.summary()))
    file.write('\n\n\n')
    file.write('Same regression after normalization and adding previous rate\n\n\n')
    RegDF = pandas.DataFrame({'const' : 1/vix[1:], 'rate' : series[:-1]/vix[1:], 'duration' : dseries/vix[1:], 'vix' : 1})
    Reg = OLS(nreturns/vix[1:], RegDF).fit()
    file.write(str(Reg.summary()))
    file.write('\n\n\n')
    
    
    file.write('Duration regression of premia minus normalized spread upon spread change\n\n\n')
    series = ratingDF['Spread'].values/100
    dseries = numpy.diff(series)
    npremia = premia - series[:-1]/12
    Reg = stats.linregress(dseries, npremia)
    s = Reg.slope
    i = Reg.intercept
    residuals = numpy.array([npremia[k] - s * dseries[k] - i for k in range(N-1)])
    file.write(str(Reg) + '\n\n')
    analysis(residuals)
    file.write('Same regression after normalization\n\n\n')
    RegDF = pandas.DataFrame({'const' : 1/vix[1:], 'duration' : dseries/vix[1:], 'vix' : 1})
    Reg = OLS(npremia/vix[1:], RegDF).fit()
    allResid.append(Reg.resid)
    file.write(str(Reg.summary()))
    file.write('\n\n\n')
    file.write('Same regression after normalization and adding previous spread\n\n\n')
    RegDF = pandas.DataFrame({'const' : 1/vix[1:], 'rate' : series[:-1]/vix[1:], 'duration' : dseries/vix[1:], 'vix' : 1})
    Reg = OLS(npremia/vix[1:], RegDF).fit()
    file.write(str(Reg.summary()))
    file.write('\n\n\n')
    
    
    file.write('Duration regression of premia minus normalized excess upon excess change\n\n\n')
    series = ratingDF['Excess'].values/100
    dseries = numpy.diff(series)
    npremia = premia - series[:-1]/12
    Reg = stats.linregress(dseries, npremia)
    s = Reg.slope
    i = Reg.intercept
    residuals = numpy.array([npremia[k] - s * dseries[k] - i for k in range(N-1)])
    file.write(str(Reg) + '\n\n')
    analysis(residuals)
    file.write('Same regression after normalization\n\n\n')
    RegDF = pandas.DataFrame({'const' : 1/vix[1:], 'duration' : dseries/vix[1:], 'vix' : 1})
    Reg = OLS(npremia/vix[1:], RegDF).fit()
    allResid.append(Reg.resid)
    file.write(str(Reg.summary()))
    file.write('\n\n\n')
    file.write('Same regression after normalization and adding previous excess\n\n\n')
    RegDF = pandas.DataFrame({'const' : 1/vix[1:], 'rate' : series[:-1]/vix[1:], 'duration' : dseries/vix[1:], 'vix' : 1})
    Reg = OLS(npremia/vix[1:], RegDF).fit()
    file.write(str(Reg.summary()))
    file.write('\n\n\n')
    
file.close()

labels = ['Yield-AR', 'Spread-AR', 'Excess-AR', 'Yield-Returns', 'Spread-Premia', 'Excess-Premia']
allLabels = labels + labels
allResid = pandas.DataFrame(columns = allLabels, data = numpy.transpose(numpy.array(allResid)))
allResid.to_excel('residuals-BofA.xlsx')