module BuyAndHold

using DataFrames
using Dates

export buy_and_hold

# const OhlcvRow = DataFrameRow{Tuple{DateTime,Float64,Float64,Float64,Float64,Int64},<:Any}

function buy_and_hold(row::DataFrameRow, ctx::Dict, positions::Dict)
  no_open_pos = get(positions, :long, 0) === 0
  is_bullish = get(ctx, :trend, "") === "bullish"
  if no_open_pos && is_bullish
    println("Opening position on $(row.Date)")
    return 1
  end
  return 0
end

end # module
