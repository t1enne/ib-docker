module Indicators

using Statistics

export sma, ema, rsi, macd

# Simple Moving Average
function sma(prices::Vector{<:Real}, period::Int)
  n = length(prices)
  result = zeros(n)
  for i in period:n
    result[i] = mean(prices[i-period+1:i])
  end
  return result
end

# Exponential Moving Average
function ema(prices::Vector{<:Real}, period::Int)
  n = length(prices)
  result = zeros(n)
  multiplier = 2 / (period + 1)
  result[1] = prices[1]
  for i in 2:n
    result[i] = (prices[i] - result[i-1]) * multiplier + result[i-1]
  end
  return result
end

# Relative Strength Index
function rsi(prices::Vector{<:Real}, period::Int=14)
  n = length(prices)
  gains = zeros(n)
  losses = zeros(n)

  for i in 2:n
    change = prices[i] - prices[i-1]
    if change > 0
      gains[i] = change
    else
      losses[i] = -change
    end
  end

  avg_gain = sma(gains, period)
  avg_loss = sma(losses, period)

  rs = zeros(n)
  rsi_values = zeros(n)

  for i in period:n
    if avg_loss[i] != 0
      rs[i] = avg_gain[i] / avg_loss[i]
      rsi_values[i] = 100 - (100 / (1 + rs[i]))
    else
      rsi_values[i] = 100
    end
  end

  return rsi_values
end

# MACD (Moving Average Convergence Divergence)
function macd(prices::Vector{<:Real}, fast_period::Int=12, slow_period::Int=26, signal_period::Int=9)
  fast_ema = ema(prices, fast_period)
  slow_ema = ema(prices, slow_period)
  macd_line = fast_ema - slow_ema
  signal_line = ema(macd_line, signal_period)
  histogram = macd_line - signal_line
  return macd_line, signal_line, histogram
end

end
