module Ibkr

using Revise, JSON, AuthStatus

export tickle

function tickle()
  gw = AuthStatus.IBKRGateway()
  r = AuthStatus.check_auth_status(gw)
  JSON.print(JSON.parse(r), 4)
end

end
