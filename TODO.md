# Migration Plan: Python to Julia for CLI Folder

## Overview
Migrate the Python scripts in the `cli` folder to Julia. This includes `auth/auth_status.py` and `candles/main.py`. The migration involves rewriting the code using Julia equivalents for Python libraries.

## Dependencies Mapping
- `requests` → `HTTP.jl`
- `urllib3` → Included in `HTTP.jl`
- `pandas` → `DataFrames.jl`
- `argparse` → `ArgParse.jl`
- `dataclasses` → Julia structs
- `datetime` → `Dates.jl`
- `typing` → Type annotations (optional in Julia)

## Step 1: Create Julia Project.toml
Create a new `Project.toml` in the `cli` folder for Julia dependencies.

```toml
[deps]
HTTP = "cd3eb016-35fb-5094-929b-558a96fad6f3"
JSON = "682c06a0-de6a-54ab-a142-c8b1cf79cde6"
DataFrames = "a93c6f00-e57d-5684-b7b6-d8193f3e46c0"
ArgParse = "c7e460c6-2fb9-53a9-8c5b-16f535851c63"
Dates = "ade2ca70-3891-5945-98fb-dc099432e06a"
```

## Step 2: Migrate auth/auth_status.jl
Rewrite `auth_status.py` to Julia.

```julia
#!/usr/bin/env julia
"""
Interactive Brokers Client Portal API - Authentication Status Example
This script demonstrates how to check the authentication status of the gateway.
"""

using HTTP
using JSON

struct IBKRGateway
    base_url::String
    api_base::String

    function IBKRGateway(base_url="https://localhost:5000")
        new(base_url, "$base_url/v1/api")
    end
end

function check_auth_status(gateway::IBKRGateway)
    endpoint = "iserver/auth/status"
    url = "$(gateway.api_base)/$endpoint"

    try
        response = HTTP.get(url, verify=false, timeout=10)
        status_data = JSON.parse(String(response.body))

        println("Authentication Status:")
        println("-" ^ 30)
        println("Authenticated: $(get(status_data, "authenticated", false))")
        println("Connected: $(get(status_data, "connected", false))")
        println("Competing: $(get(status_data, "competing", false))")
        println("Message: $(get(status_data, "message", "N/A"))")

        if haskey(status_data, "serverInfo")
            println("Server Info: $(status_data["serverInfo"])")
        end

        return status_data
    catch e
        println("Error checking authentication status: $e")
        return nothing
    end
end

function tickle(gateway::IBKRGateway)
    endpoint = "tickle"
    url = "$(gateway.api_base)/$endpoint"

    try
        response = HTTP.post(url, verify=false, timeout=10)
        println("Session tickle successful")
        return true
    catch e
        println("Error sending tickle: $e")
        return false
    end
end

function main()
    gateway = IBKRGateway()

    println("IBKR Client Portal Gateway - Authentication Status Check")
    println("=" ^ 55)

    status = check_auth_status(gateway)

    if status !== nothing
        if get(status, "authenticated", false) && get(status, "connected", false)
            println("\n✅ Gateway is authenticated and connected!")
            # Uncomment to send tickle
            # println("\nSending session tickle...")
            # tickle(gateway)
        else
            println("\n❌ Gateway is not properly authenticated.")
            println("Please visit https://localhost:5000 to login.")
        end
    else
        println("\n❌ Unable to connect to gateway.")
        println("Make sure the gateway is running on https://localhost:5000")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
```

## Step 3: Migrate candles/main.jl
Rewrite `main.py` to Julia.

```julia
#!/usr/bin/env julia
"""
Interactive Brokers Client Portal API - Candlestick Data Fetcher
This script fetches OHLCV candlestick data for specified symbols and time ranges.
"""

using ArgParse
using DataFrames
using Dates
using HTTP
using JSON

struct Security
    conid::Int
    symbol::String
    sections::Vector{Dict{String, String}}
    secType::String
    description::String
    company_name::String
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
        response = HTTP.get(url, query=params, verify=false, timeout=10)
        data = JSON.parse(String(response.body))

        if haskey(data, "error") && data["error"] !== nothing
            println("Failed to fetch symbols: $(data["error"])")
            return []
        end

        if isempty(data)
            return []
        end

        securities = []
        for item in data
            if haskey(item, "secType")
                sec_type = item["secType"]
            elseif haskey(item, "sections") && !isempty(item["sections"]) && haskey(item["sections"][1], "secType")
                sec_type = item["sections"][1]["secType"]
            else
                throw(ErrorException("Unrecognized security type"))
            end

            security = Security(
                get(item, "conid", 0),
                get(item, "symbol", ""),
                get(item, "sections", []),
                sec_type,
                get(item, "description", ""),
                get(item, "companyHeader", "")
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
        println("$i. *$(sec.symbol)* $(sec.company_name) _$(sec.secType)_ ($(sec.description)) [$(sec.conid)]")
    end

    while true
        print("\nSelect security (1-$(length(securities))): ")
        choice = readline()
        try
            idx = parse(Int, choice) - 1
            if 0 <= idx < length(securities)
                return securities[idx]
            else
                println("Invalid selection. Please try again.")
            end
        catch
            println("\nOperation cancelled.")
            return nothing
        end
    end
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
        response = HTTP.get(url, query=params, verify=false, timeout=30)
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

function main()
    s = ArgParseSettings(description="Fetch OHLCV candlestick data from IBKR")
    @add_arg_table s begin
        "symbol"
            help = "Stock symbol to fetch data for"
            required = true
        "--interval", "-i"
            help = "Time interval (default: 1d)"
            default = "1d"
        "--start-date", "-s"
            help = "Start date (YYYY-MM-DD)"
        "--end-date", "-e"
            help = "End date (YYYY-MM-DD)"
        "--sec-type", "-t"
            help = "Security type (STK, OPT, BOND, etc.)"
        "--output", "-o"
            help = "Output CSV dir"
        "--period", "-p"
            help = "Time away from startTime. period=6d"
    end

    args = parse_args(s)
    fetcher = CandlesFetcher()

    println("Searching for $(args["symbol"])...")

    securities = search_securities(fetcher, args["symbol"])
    contract = get_contract_id(securities, args["sec-type"])
    if contract === nothing
        println("Unable to get contract for symbol: $(args["symbol"])")
        return
    end

    df = fetch_candles(fetcher, contract.conid, args["interval"], args["period"], args["start-date"], args["end-date"])

    if df !== nothing && !isempty(df)
        println("\nFetched $(nrow(df)) candles:")
        println(df)
        filename = "$(args["symbol"])_$(contract.secType)_$(args["interval"])"
        filepath = "$(args["output"])/$filename.csv"

        if args["output"] !== nothing
            CSV.write(filepath, df)
            println("\nData saved to: $filepath")
        end
    else
        println("No data retrieved")
        exit(1)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
```

## Step 4: Update README Files
Update `cli/auth/README.md` and `cli/candles/README.md` to include Julia setup instructions.

For example, in `cli/auth/README.md`:
```
# Authentication Status Checker

This Julia script checks the authentication status of the IBKR Client Portal Gateway.

## Setup
1. Install Julia 1.6+
2. Activate the project: `cd cli && julia --project=. -e 'using Pkg; Pkg.instantiate()'`
3. Run: `julia --project=. auth/auth_status.jl`
```

## Step 5: Testing
- Test the migrated scripts with the IBKR gateway.
- Ensure all functionality matches the Python versions.
- Handle any SSL certificate issues as in the original code.