from __future__ import annotations

def guess_signal_ticker(risk_ticker: str) -> str:
    """
    Map leveraged products to a reasonable unlevered signal source.
    You can override in UI/API with a custom signal ticker.
    """
    t = (risk_ticker or "").upper().strip()

    mapping = {
        # 3x/2x Nasdaq
        "TQQQ": "QQQ",
        "SQQQ": "QQQ",

        # 3x/2x S&P
        "UPRO": "SPY",
        "SPXL": "SPY",
        "SPXU": "SPY",
        "SSO":  "SPY",
        "SDS":  "SPY",

        # 3x/2x Dow
        "UDOW": "DIA",
        "SDOW": "DIA",
        "DDM":  "DIA",

        # 3x/2x Russell
        "TNA":  "IWM",
        "TZA":  "IWM",
        "UWM":  "IWM",

        # 3x bonds
        "TMF":  "TLT",
        "TMV":  "TLT",

        # common tech / broad
        "QLD":  "QQQ",
        
        #Semiconductor
        "TSXU":  "QQQ",
        "TTXU":  "QQQ",
        
        #Individuyal Stocks
        "NVDU":  "NVDA",
        "AAPU":  "AAPL",
        "AMDU":  "AMD",
        "AMZU":  "AMZN",
        "AVL":  "AVGO",
        "BRKI":  "BRKB",
        "GGLL":  "GOOGL",
        "METU":  "META",
        "MSFU":  "MSFT",
        "MUU":  "MU",
        "ELIL":  "LLY",
        "PALU":  "PLTR",
        "TSLL":  "TSLA",
        "TSMX":  "TSM",
    }

    return mapping.get(t, t)  # default: same ticker


def supported_main_tickers() -> list[str]:
    """Used by Streamlit dropdown."""
    keys = [
        "TQQQ","SQQQ","QLD",
        "UPRO","SPXL","SPXU","SSO","SDS",
        "UDOW","SDOW","DDM",
        "TNA","TZA","UWM",
        "TMF","TMV", "QLD", "TSXU", "TTXU",
        "NVDU", "AAPU", "AMDU",
        "AMZU", "AVL", 
        "BRKI",
        "GGLL",
        "METU",
        "MSFU",
        "MUU",
        "ELIL",
        "PALU",
        "TSLL",
        "TSMX",
    ]
    return sorted(set(keys))
