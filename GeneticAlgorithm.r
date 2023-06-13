# Data preparation

f = NULL
files = c("AAPL.csv","AMZN.csv","GOOG.csv", "META.csv", "NDAQ.csv")
for (i in 1:length(files)) {
  csv = read.csv(files[i])
  csv = csv[,c("Date","Close")]
  names(csv) = c("Date",substr(files[i], 36, nchar(files[i])))
  if (i == 1) f = csv
  else f = merge(f,csv)
}

f

for (i in 2:ncol(f)) {
  # Price time series of the i-th asset
  prices = f[,i] 
  
  # Price lagged by 1
  prices_prev = c(NA,prices[1:(length(prices)-1)]) 
  
  # Returns time series
  returns = (prices-prices_prev)/prices_prev 
  
  # Replace the i-th column with returns
  f[,i] = returns 
}
# Remove the first row with NAs and the Date column
asset_returns = f[2:nrow(f),2:ncol(f)]

asset_returns

portfolio_returns = function(x) {
  port.returns = 0
  
  # Multiplication of the i-th asset by the i-th weight in "x"
  for (i in 1:length(x)) {
    port.returns = port.returns + asset_returns[,i] * x[i]
  }
  
  return (port.returns)
}

sharpe = function(x) {
  port.returns = portfolio_returns(x)
  
  return (mean(port.returns)/sqrt(var(port.returns)))
  
}

constraint = function(x) {
  boundary_constr = (sum(x)-1)**2   # "sum x = 1" constraint
  
  for (i in 1:length(x)) {
    boundary_constr = boundary_constr + 
      max(c(0,x[i]-1))**2 +  # "x <= 1" constraint
      max(c(0,-x[i]))**2     # "x >= 0" constraint
  }
  
  return (boundary_constr)
}


obj = function(x) {
  # We want the maximum Sharpe ratio, so we multiply it by
  # -1 to fit an optimization problem
  
  return (-sharpe(x)+100*constraint(x))
}


library("GA")
ga_res = ga(
  # Tell the genetic algorithm that the 
  # weights are real variables
  type="real-valued", 
  
  # "ga" function performs maximization, so we must
  # multiply the objective function by -1
  function(x){-obj(x)}, 
  
  # x_i >= 0
  lower = rep(0,ncol(asset_returns)), 
  
  # x_i <= 1
  upper = rep(1,ncol(asset_returns)), 
  
  # Maximum number of iterations 
  maxiter = 50000, 
  
  # If the maximum fitness remains the same for 50
  # consecutive transactions, stop the algorithm
  run=60, 
  
  # Exploit multi-core properties of your CPU
  parallel=TRUE,
  
  # We want to see the partial results of the process
  # while it performs
  monitor=TRUE,
  
  # Seed useful for replicating the results
  seed=1
)

sol = as.vector(summary(ga_res)$solution)

optimal_returns = portfolio_returns(sol)
plot(cumsum(optimal_returns),type="l",lwd=5)
lines(cumsum(asset_returns[,1]),col="blue")
lines(cumsum(asset_returns[,2]),col="red")
lines(cumsum(asset_returns[,3]),col="green")
lines(cumprod(asset_returns[,4]),col="violet")
lines(cumsum(asset_returns[,5]),col="peru")
legend(0,1.5,legend=c("Weighted portfolio",names(asset_returns)),
       col = c("black","blue","red","green","violet","peru"),lty=1)

cbind(names(asset_returns), sol)
