module AuthStatus

using HTTP
using JSON

export IBKRGateway, check_auth_status, tickle

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
    response = HTTP.get(url; require_ssl_verification=false, timeout=10)
    status_data = JSON.parse(String(response.body))

    println("Authentication Status:")
    println("-"^30)
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
    HTTP.post(url; require_ssl_verification=false, timeout=10)
    println("Session tickle successful")
    return true
  catch e
    println("Error sending tickle: $e")
    return false
  end
end

end
