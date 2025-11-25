import { ISymbol } from "../db/types";

interface Props {
  symbol: ISymbol;
}
export function Symbol({ symbol }: Props) {
  return (
    <div className="grid grid-cols-3">
      <div className="hidden">{symbol.id}</div>
      <div>{symbol.name}</div>
      <div>{symbol.market}</div>
      <div>{symbol.currency}</div>
    </div>
  );
}
