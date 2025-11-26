import { ISecurity } from "../types/ibkr";

interface Props {
  symbol: ISecurity;
}
export function AddableSymbol({ symbol }: Props) {
  return (
    <div class="border p-2 mb-2">
      <p>
        <strong>{symbol.symbol}</strong> - {symbol.companyName}
      </p>
      <p>{symbol.description}</p>
      <button
        type="button"
        hx-post={`/symbols/${symbol.conid}`}
        hx-swap="innerHTML"
        class="bg-purple-500 text-white px-2 py-1 text-sm"
      >
        Add
      </button>
    </div>
  );
}
