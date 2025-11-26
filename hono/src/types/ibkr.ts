export interface SymbolInfo {
  cfi_code: string; // "";
  symbol: string; // "IBM";
  cusip: unknown;
  expiry_full: unknown;
  con_id: number;
  maturity_date: unknown;
  industry: string; // "Computers";
  instrument_type: string; // "STK";
  trading_class: string; // "IBM";
  valid_exchanges: string; // "SMART,AMEX,NYSE,CBOE,PHLX,ISE,CHX,ARCA,NASDAQ,DRCTEDGE,BEX,BATS,EDGEA,BYX,NYSEDARK,NASDDARK,IEX,EDGX,FOXRIVER,PEARL,NYSENAT,IEXMID,JANELP,IMCLP,LTSE,MEMX,JUMPLP,OLDMCLP,RBCCMALP,IBEOS,GSLP,BLUEOCEAN,OVERNIGHT,JANEMID,G1XLP,PSX";
  allow_sell_long: boolean;
  is_zero_commission_security: boolean;
  local_symbol: string; // "IBM";
  contract_clarification_type: unknown;
  classifier: unknown;
  currency: string; // "USD";
  text: unknown;
  underlying_con_id: number;
  r_t_h: boolean;
  multiplier: unknown;
  underlying_issuer: unknown;
  contract_month: unknown;
  company_name: string; // "INTL BUSINESS MACHINES CORP";
  smart_available: boolean;
  exchange: string; // "SMART";
  category: string; // "Computer Services";
}

export interface IWatchlist {
  id: string;
  is_open: boolean;
  read_only: boolean;
  name: string;
  modified: number;
  type: string;
}

export interface IWatchedSecurity {
  ST: string; // "STK";
  C: string; // "81547099";
  conid: number; // same as "C"
  name: string; // "PANDORA A/S";
  fullName: string; // "PNDORA";
  assetClass: string; // "STK";
  ticker: string; // "PNDORA";
}

export interface ISecurity {
  conid: string; // "8314";
  companyHeader: string; // "INTL BUSINESS MACHINES CORP - NYSE";
  companyName: string; // "INTL BUSINESS MACHINES CORP";
  symbol: string; //  "IBM";
  description: string; //  "NYSE";
  restricted: null;
  sections: Array<{ secType: string; exchange?: string }>;
}
