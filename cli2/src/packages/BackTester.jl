using DataFrames, CSV, Dates

include("../strategies/SmaCross.jl")
using .SmaCross

include("../strategies/BuyAndHold.jl")
using .BuyAndHold

export run_backtest

function run_backtest(data_file::String, strat_sym::Symbol)
  println("Testing $strat_sym")
  # Load historical data
  df = CSV.read(data_file, DataFrame)
  strategies_map = Dict(
    :sma_cross => sma_cross,
    :buy_and_hold => buy_and_hold
  )

  # Ensure Date column is DateTime
  df.Date = DateTime.(df.Date, "yyyy-mm-dd HH:MM:SS")

  # Precompute indicators if needed
  if strat_sym == :sma_cross
    df.short_sma = sma(df.Close, 10)
    df.long_sma = sma(df.Close, 20)
  end

  capital = 10000.0
  allocation_pct = 0.2  # 10% of equity per trade
  position = 0  # 0: no position, 1: long
  shares = 0
  entry_price = 0.0
  trades = 0
  wins = 0

  # For plotting
  prices = Float64[]
  volumes = Float64[]
  equities = Float64[]
  buy_signals = Int[]
  sell_signals = Int[]

  ctx = Dict(:trend => "bullish")

  strat = strategies_map[strat_sym]

  for (index, row) in enumerate(eachrow(df))
    sig = strat(row, ctx, Dict(:long => position > 0 ? 1 : 0, :short => 0))
    push!(prices, row.Close)
    push!(volumes, row.Volume)
    push!(equities, capital)

    if sig == 1 && position == 0
      # Buy
      amount_to_invest = capital * allocation_pct
      shares_to_buy = floor(amount_to_invest / row.Close)
      if shares_to_buy > 0
        cost = shares_to_buy * row.Close
        capital -= cost
        position = 1
        shares = shares_to_buy
        entry_price = row.Close
        trades += 1
        push!(buy_signals, index)
        push!(sell_signals, 0)
      else
        push!(buy_signals, 0)
        push!(sell_signals, 0)
      end
    elseif sig == -1 && position == 1
      # Sell
      proceeds = shares * row.Close
      pnl = proceeds - (shares * entry_price)
      capital += proceeds
      if pnl > 0
        wins += 1
      end
      position = 0
      shares = 0
      push!(sell_signals, index)
      push!(buy_signals, 0)
    else
      push!(buy_signals, 0)
      push!(sell_signals, 0)
    end
  end

  # Close any open position
  if position == 1
    proceeds = shares * df.Close[end]
    pnl = proceeds - (shares * entry_price)
    capital += proceeds
    if pnl > 0
      wins += 1
    end
    trades += 1
  end

  win_rate = trades > 0 ? wins / trades : 0.0

  println("Final Capital: \$$(round(capital, digits=2))")
  println("Total Trades: $trades")
  println("Win Rate: $(round(win_rate * 100, digits=2))%")
  println("Total Return: $(round((capital - 10000) / 10000 * 100, digits=2))%")
  println("===================")

  # Plot results
  # plot_results(df.Date, prices, volumes, equities, buy_signals, sell_signals)
end



# function plot_results(dates::Vector{DateTime}, prices::Vector{Float64}, volumes::Vector{Float64}, equities::Vector{Float64}, buy_signals::Vector{Int}, sell_signals::Vector{Int})
#    fig = Figure(size=(1200, 800))
#
#    # Price chart with trades
#    ax1 = Axis(fig[1,1], title="Price and Trades", ylabel="Price")
#    lines!(ax1, dates, prices, color=:blue, linewidth=2)
#
#    buy_indices = findall(x -> x > 0, buy_signals)
#    sell_indices = findall(x -> x > 0, sell_signals)
#
#    if !isempty(buy_indices)
#      scatter!(ax1, dates[buy_indices], prices[buy_indices], color=:green, marker=:utriangle, markersize=10)
#    end
#    if !isempty(sell_indices)
#      scatter!(ax1, dates[sell_indices], prices[sell_indices], color=:red, marker=:dtriangle, markersize=10)
#    end
#
#    # Volume chart
#    ax2 = Axis(fig[2,1], title="Volume", ylabel="Volume")
#    barplot!(ax2, dates, volumes, color=:gray)
#
#    # Equity chart
#    ax3 = Axis(fig[3,1], title="Equity Curve", ylabel="Equity", xlabel="Date")
#    lines!(ax3, dates, equities, color=:purple, linewidth=2)
#
#    # Link x-axes for zooming
#    linkxaxes!(ax1, ax2, ax3)
#
#    display(fig)
#    println("Plot displayed")
#  end

