module IbkrCli

using ArgParse
using HTTP
using JSON
using DataFrames
using Dates
using CSV

include("./AuthStatus.jl")
include("./Candles.jl")
include("./BackTester.jl")

using .AuthStatus
using .Candles
using .BackTester

export main

function main()
    s = ArgParseSettings(description="Interactive Brokers CLI Tool")
    @add_arg_table s begin
        "command"
            help = "Subcommand: auth or candles"
            required = true
        "subargs"
            help = "Arguments for subcommand"
            nargs = '*'
    end

    args = parse_args(s)

		if args["command"] == "bt"
			backtest()	
			return
		end

		if args["command"] == "auth"
			if length(args["subargs"]) >= 1 && args["subargs"][1] == "status"
				url = length(args["subargs"]) >= 2 ? args["subargs"][2] : "https://localhost:5000"
				gateway = IBKRGateway(url)
				status = check_auth_status(gateway)
				if status !== nothing
					if get(status, "authenticated", false) && get(status, "connected", false)
						println("✅ Gateway is authenticated and connected!")
					else
						println("❌ Gateway is not properly authenticated.")
					end
				else
					println("❌ Unable to connect to gateway.")
				end
			else
				println("Usage: ibcli auth status [url]")
			end
		elseif args["command"] == "candles"
			if isempty(args["subargs"])
				println("Usage: ibcli candles <symbol> [options]")
				return
			end

			symbol = args["subargs"][1]
			interval = "1d"
			period = "30d"
			sec_type = nothing
			output = nothing

			i = 2
			while i <= length(args["subargs"])
				if args["subargs"][i] == "-i" && i + 1 <= length(args["subargs"])
					interval = args["subargs"][i + 1]
					i += 2
				elseif args["subargs"][i] == "-p" && i + 1 <= length(args["subargs"])
					period = args["subargs"][i + 1]
					i += 2
				elseif args["subargs"][i] == "-t" && i + 1 <= length(args["subargs"])
					sec_type = args["subargs"][i + 1]
					i += 2
				elseif args["subargs"][i] == "-o" && i + 1 <= length(args["subargs"])
					output = args["subargs"][i + 1]
					i += 2
				else
					i += 1
				end
			end

			fetcher = CandlesFetcher()
			println("Searching for $symbol...")

			securities = search_securities(fetcher, symbol)
			contract = get_contract_id(securities, sec_type)
			if contract === nothing
				println("Unable to get contract for symbol: $symbol")
				return
			end

			df = fetch_candles(fetcher, contract.conid !== nothing ? contract.conid : 0, interval, period)

			if df !== nothing && !isempty(df)
				println("\nFetched $(nrow(df)) candles:")
				println(df)
				if output !== nothing
					CSV.write(output, df)
					println("\nData saved to: $output")
				end
			else
				println("No data retrieved")
			end
		else
			println("Unknown command: $(args["command"])")
			println("Available commands: auth, candles")
		end
	end

end # module IBCLI
