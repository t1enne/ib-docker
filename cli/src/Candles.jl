module Candles

using DataFrames
using Dates
using HTTP
using JSON
using CSV

export Security, CandlesFetcher, search_securities, get_contract_id, fetch_candles

struct Security
    conid::Union{Int, Nothing}
    symbol::Union{String, Nothing}
    sections::Union{Vector{Dict{String, Any}}, Nothing}
    secType::Union{String, Nothing}
    description::Union{String, Nothing}
    company_name::Union{String, Nothing}
end

struct CandlesFetcher
    base_url::String
    api_base::String

    function CandlesFetcher(base_url="https://localhost:5000")
        new(base_url, "$base_url/v1/api")
    end
end

function search_securities(fetcher::CandlesFetcher, symbol::String)::Vector{Security}
    endpoint = "iserver/secdef/search"
    url = "$(fetcher.api_base)/$endpoint"
    params = Dict("symbol" => symbol)

    try
        response = HTTP.get(url; query=params, require_ssl_verification=false, timeout=10)
        data = JSON.parse(String(response.body))

        if isa(data, Array) && !isempty(data) && last(data) == "error"
            println("Failed to fetch symbols: API returned error")
            return []
        end

        if isempty(data)
            return []
        end

        securities = Security[]
        for item in data
            if haskey(item, "secType")
                sec_type = item["secType"]
            elseif haskey(item, "sections") && !isempty(item["sections"])
                sec_type = item["sections"][1]["secType"]
            else
                throw(ErrorException("Unrecognized security type"))
            end

            conid_str = get(item, "conid", nothing)
            conid = conid_str !== nothing ? tryparse(Int, string(conid_str)) : nothing

            security = Security(
                conid,
                get(item, "symbol", nothing),
                get(item, "sections", nothing),
                sec_type,
                get(item, "description", nothing),
                get(item, "companyHeader", nothing)
            )
            push!(securities, security)
        end

        return securities
    catch e
        println("Error searching for $symbol: $e")
        return []
    end
end

function get_contract_id(securities::Vector{Security}, sec_type::Union{String, Nothing})::Union{Security, Nothing}
		println(securities)
    if isempty(securities)
        println("No securities found")
        return nothing
    end

    if sec_type !== nothing
        filtered = filter(s -> s.secType == sec_type, securities)
        if !isempty(filtered)
            return filtered[1]
        end
    end

    if length(securities) == 1
        return securities[1]
    end

    println("\nFound $(length(securities)) securities:")
    println("-" ^ 50)
    for (i, sec) in enumerate(securities)
        println("$i. *$(sec.symbol !== nothing ? sec.symbol : "N/A")* $(sec.company_name !== nothing ? sec.company_name : "N/A") _$(sec.secType !== nothing ? sec.secType : "N/A")_ ($(sec.description !== nothing ? sec.description : "N/A")) [$(sec.conid !== nothing ? sec.conid : "N/A")]")
    end

    # Non-interactive mode, select the first one
    return securities[1]
end

function fetch_candles(fetcher::CandlesFetcher, conid::Int, interval::String="1d", period::String="30d", start_date::Union{String, Nothing}=nothing, end_date::Union{String, Nothing}=nothing)::Union{DataFrame, Nothing}
    endpoint = "iserver/marketdata/history"
    url = "$(fetcher.api_base)/$endpoint"
    params = Dict("conid" => string(conid), "period" => period, "bar" => interval)

    if start_date !== nothing
        start_ts = Int64(Dates.datetime2unix(DateTime(start_date, "yyyy-mm-dd")) * 1000)
        params["startTime"] = string(start_ts)
    end

    if end_date !== nothing
        end_ts = Int64(Dates.datetime2unix(DateTime(end_date, "yyyy-mm-dd")) * 1000)
        params["endTime"] = string(end_ts)
    end

    try
        response = HTTP.get(url; query=params, require_ssl_verification=false, timeout=30)
        data = JSON.parse(String(response.body))

        if !haskey(data, "data")
            println("No data returned for $conid")
            return nothing
        end

        candles_data = data["data"]
        df = DataFrame(
            Date = [Dates.format(Dates.unix2datetime(candle["t"] / 1000), "yyyy-mm-dd HH:MM:SS") for candle in candles_data],
            Open = [get(candle, "o", 0.0) for candle in candles_data],
            High = [get(candle, "h", 0.0) for candle in candles_data],
            Low = [get(candle, "l", 0.0) for candle in candles_data],
            Close = [get(candle, "c", 0.0) for candle in candles_data],
            Volume = [get(candle, "v", 0) for candle in candles_data]
        )

        return df
    catch e
        println("Error fetching candles for $conid: $e")
        return nothing
    end
end

end # module
