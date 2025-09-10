using DataFrames

# Simple Moving Average Crossover Strategy
# Buy when short SMA crosses above long SMA
# Sell when short SMA crosses below long SMA

export sma_cross

function sma_cross(row::DataFrameRow, ctx::Dict, positions::Dict)
  short_period = 10
  long_period = 20

  # Assume SMAs are precomputed in df
  if hasproperty(row, :short_sma) && hasproperty(row, :long_sma) && !ismissing(row.short_sma) && !ismissing(row.long_sma)
    prev_short = get(ctx, :prev_short_sma, row.short_sma)
    prev_long = get(ctx, :prev_long_sma, row.long_sma)

    if row.short_sma > row.long_sma && prev_short <= prev_long
      ctx[:prev_short_sma] = row.short_sma
      ctx[:prev_long_sma] = row.long_sma
      return 1
    elseif row.short_sma < row.long_sma && prev_short >= prev_long
      ctx[:prev_short_sma] = row.short_sma
      ctx[:prev_long_sma] = row.long_sma
      return -1
    end

    ctx[:prev_short_sma] = row.short_sma
    ctx[:prev_long_sma] = row.long_sma
  end

  return 0
end

