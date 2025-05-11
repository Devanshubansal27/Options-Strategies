import yfinance as yf
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware with more permissive settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
    max_age=3600,
)

# Define standard lot size used in options contracts
LOT_SIZE = 100

def format_usd(value):
    """Format a numeric value into USD string with + or - sign."""
    if value is None:
        return "$N/A"
    if value > 0:
        return f"+${round(value, 2)}"
    elif value < 0:
        return f"-${abs(round(value, 2))}"
    else:
        return "$0.00"

class StrategyBase:
    """
    Base class for all options strategies.

    Attributes:
        price (float): Underlying price at expiry.
        atm_strike (float): At-the-money strike price.
        current_price (float): Current price of the stock.
        calls (DataFrame): Option chain call data.
        puts (DataFrame): Option chain put data.
    """
    def __init__(self, price, atm_strike, current_price, calls, puts):
        self.price = price
        self.atm_strike = atm_strike
        self.current_price = current_price
        self.calls = calls
        self.puts = puts

    def get_price(self, df, strike):
        """Get last traded price for a specific strike."""
        row = df[df['strike'] == strike]
        if not row.empty:
            return round(row.iloc[0]['lastPrice'], 2)
        raise IndexError("Strike not found in the option chain.")

class LongCall(StrategyBase):
    """Profit from a rise in the underlying stock above the strike price."""
    def calculate(self):
        call_atm = self.get_price(self.calls, self.atm_strike)
        return (max(self.price - self.atm_strike, 0) - call_atm) * LOT_SIZE

class LongPut(StrategyBase):
    """Profit from a drop in the underlying stock below the strike price."""
    def calculate(self):
        put_atm = self.get_price(self.puts, self.atm_strike)
        return (max(self.atm_strike - self.price, 0) - put_atm) * LOT_SIZE

class CoveredCall(StrategyBase):
    """Combines holding the stock with selling a call to generate premium income."""
    def calculate(self):
        call_atm = self.get_price(self.calls, self.atm_strike)
        return ((self.price - self.current_price) + call_atm - max(self.price - self.atm_strike, 0)) * LOT_SIZE

class ProtectivePut(StrategyBase):
    """Combines holding the stock with buying a put for downside protection."""
    def calculate(self):
        put_atm = self.get_price(self.puts, self.atm_strike)
        return ((self.price - self.current_price) - put_atm + max(self.atm_strike - self.price, 0)) * LOT_SIZE

class Straddle(StrategyBase):
    """Buy a call and a put at the same strike, expecting high volatility."""
    def calculate(self):
        lc = LongCall(self.price, self.atm_strike, self.current_price, self.calls, self.puts).calculate()
        lp = LongPut(self.price, self.atm_strike, self.current_price, self.calls, self.puts).calculate()
        return lc + lp

class Strangle(StrategyBase):
    """Buy a call and a put with different OTM strikes, expecting large movement."""
    def calculate(self):
        call_otm = self.get_price(self.calls, round(self.atm_strike + 5, 2))
        put_otm = self.get_price(self.puts, round(self.atm_strike - 5, 2))
        return (max(self.price - (self.atm_strike + 5), 0) +
                max((self.atm_strike - 5) - self.price, 0) - call_otm - put_otm) * LOT_SIZE

class BullCallSpread(StrategyBase):
    """Buy a call and sell a higher strike call to limit upside risk."""
    def calculate(self):
        call_atm = self.get_price(self.calls, self.atm_strike)
        call_otm = self.get_price(self.calls, round(self.atm_strike + 10, 2))
        return (max(self.price - self.atm_strike, 0) - max(self.price - (self.atm_strike + 10), 0)
                - (call_atm - call_otm)) * LOT_SIZE

class BearPutSpread(StrategyBase):
    """Buy a put and sell a lower strike put to limit downside risk."""
    def calculate(self):
        put_atm = self.get_price(self.puts, self.atm_strike)
        put_lower = self.get_price(self.puts, round(self.atm_strike - 10, 2))
        return (max(self.atm_strike - self.price, 0) - max((self.atm_strike - 10) - self.price, 0)
                - (put_atm - put_lower)) * LOT_SIZE

class IronCondor(StrategyBase):
    """Combine bull put and bear call spreads to profit from low volatility."""
    def calculate(self):
        put_inner = self.get_price(self.puts, round(self.atm_strike - 5, 2))
        put_outer = self.get_price(self.puts, round(self.atm_strike - 10, 2))
        call_inner = self.get_price(self.calls, round(self.atm_strike + 5, 2))
        call_outer = self.get_price(self.calls, round(self.atm_strike + 10, 2))
        return (-max(round(self.atm_strike - 5, 2) - self.price, 0) +
                max(round(self.atm_strike - 10, 2) - self.price, 0) -
                max(self.price - round(self.atm_strike + 5, 2), 0) +
                max(self.price - round(self.atm_strike + 10, 2), 0) +
                put_inner - put_outer + call_inner - call_outer) * LOT_SIZE

class ButterflySpread(StrategyBase):
    """Combine three call positions to limit both risk and reward around the ATM strike."""
    def calculate(self):
        call_atm = self.get_price(self.calls, self.atm_strike)
        call_lower = self.get_price(self.calls, round(self.atm_strike - 5, 2))
        call_upper = self.get_price(self.calls, round(self.atm_strike + 5, 2))
        return (max(self.price - (self.atm_strike - 5), 0) -
                2 * max(self.price - self.atm_strike, 0) +
                max(self.price - (self.atm_strike + 5), 0) -
                (call_lower + call_upper - 2 * call_atm)) * LOT_SIZE

@app.get("/options-strategy-pnl")
def get_strategy_pnl(
    ticker: str = Query(..., description="Stock symbol (e.g., AAPL)")
):
    """
    API endpoint that calculates option strategy P&L for the given ticker.
    Automatically selects the nearest expiry.

    Returns:
        JSON object containing:
        - ticker
        - current price
        - ATM strike used
        - List of strategies with P&L at various expiry prices
    """
    try:
        stock = yf.Ticker(ticker.upper())
        current_price = stock.history(period='1d')['Close'].iloc[-1]

        expiry = stock.options[0]  # Select the nearest expiry by default
        opt_chain = stock.option_chain(expiry)
        calls = opt_chain.calls
        puts = opt_chain.puts

        valid_strikes = sorted(set(calls['strike']).intersection(set(puts['strike'])))
        if not valid_strikes:
            return JSONResponse({"error": "No overlapping strikes found."}, status_code=400)

        atm_strike = min(valid_strikes, key=lambda x: abs(x - current_price))
        price_points = np.round(np.linspace(atm_strike * 0.9, atm_strike * 1.1, 10), 2)

        results = []

        for price in price_points:
            row = {'Price at Expiry': f"${round(price, 2)}"}
            for cls in [LongCall, LongPut, CoveredCall, ProtectivePut, Straddle, Strangle, BullCallSpread, BearPutSpread, IronCondor, ButterflySpread]:
                try:
                    pnl = cls(price, atm_strike, current_price, calls, puts).calculate()
                    row[cls.__name__.replace("Strategy", "")] = format_usd(pnl)
                except Exception:
                    row[cls.__name__.replace("Strategy", "")] = "$N/A"

            results.append(row)

        df = pd.DataFrame(results)
        df = df.fillna("N/A")

        return {
            "ticker": ticker.upper(),
            "current_price": round(current_price, 2),
            "atm_strike_used": atm_strike,
            "expiry": expiry,
            "strategies": df.to_dict(orient="records")
        }

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

if __name__ == "__main__":
    import uvicorn
    # For local testing: python main.py
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
