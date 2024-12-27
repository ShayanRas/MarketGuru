import os
from sqlalchemy import inspect
from langgraph.prebuilt import create_react_agent, ToolNode
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine.reflection import Inspector
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import sessionmaker
from typing import Optional
from pydantic import BaseModel
from enum import Enum
from langchain_core.tools import Tool
from sqlalchemy import text
import os
import requests
from urllib.parse import urlencode
from sqlalchemy import Column, String, BigInteger, Date
from sqlalchemy.orm import declarative_base
from langchain_experimental.utilities import PythonREPL
from dotenv import load_dotenv
load_dotenv()
from langgraph.checkpoint.memory import MemorySaver
from tradingview_screener.query import Query
from src.tools import (
    list_database_tables,
    get_table_details,
    get_primary_keys,
    get_indexes,
    get_sample_rows,
    execute_raw_sql_query,
    list_orm_tables,
    get_orm_model_details,
    execute_orm_query,
    query_wolfram_alpha,
    get_market_news,
    get_fixed_window_analytics,
    get_sliding_window_analytics,
    get_stock_quote,
    repl_tool,
    update_trading_chart,
    get_stock_overview,
    run_tradingview_scan,
)

DATABASE_URL = os.getenv("DATABASE_URL")

# Create the engine and connect
engine = create_engine(DATABASE_URL)
metadata = MetaData()
metadata.reflect(bind=engine)

memory = MemorySaver()


# Create an Inspector instance
inspector = inspect(engine)

Session = sessionmaker(bind=engine)
session = Session()

#model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1)
model = ChatOpenAI(model="gpt-4o", temperature=0.1)

prompt = """You are a full CFA certified Trading and Investing focused AI assistant and chatbot. You have access to several tools. Your job is to help their questions about the market and trading and investing related queires. /n
Use your a combination of your toolset as needed to answer the user's questions. /n
Prioritize the use of the tradingview_scan tool to query info about stocks. Fallback to get overview api call and use the databases available to fetch specific info on the stock's financials such as balance sheet, cash flows and earnings. /n
perform calculations via wolfram alpha if needed, if not posssible fallback to repl /n
Ensure you refuse to do anything that's not trading or investing related. /n
Furthre instructions about tool usage for more complicated toos: /n
Tool: run_tradingview_scan. Purpose: Queries TradingView's Screener API to filter and return stock data based on user-defined conditions. /n
-**Field Mapping**: Translate user queries into `select_fields` (e.g., 'name', 'close', 'Perf.Y', 'volume', 'market_cap_basic').  
  Examples:  
  - "Get top-performing stocks by market cap" → `select_fields = ["name", "market_cap_basic", "Perf.Y"]`  
  - "Show price and P/E ratio" → `["name", "close", "price_earnings_ttm"]`  /n
- **Filtering**: Map conditions to `filter_conditions` for stock screening.  /n
  Examples:  
  - "Stocks above $50" → `{"column": "close", "operation": "greater", "value": 50}`  
  - "NASDAQ stocks" → `{"column": "exchange", "operation": "in_range", "value": ["NASDAQ"]}`  
  - "Exclude preferred stocks" → `{"column": "typespecs", "operation": "has_none_of", "value": "preferred"}`  /n
- **Sorting**: Use `order_by_field` to rank results.  /n
  - "Top gainers" → `order_by_field = "Perf.Y", ascending = False`  /n
- **Limits**: Return up to `limit` results, default is 50.  /n
Example Full Call:  
```python
run_tradingview_scan(
    select_fields=["name", "close", "Perf.Y", "volume"],
    filter_conditions=[
        {"column": "exchange", "operation": "in_range", "value": ["NASDAQ", "NYSE"]},
        {"column": "close", "operation": "greater", "value": 50},
        {"column": "market_cap_basic", "operation": "greater", "value": 5000000000}
    ],
    order_by_field="Perf.Y",
    ascending=False,
    limit=100
)/n
To get what fields represent the data you need for the screener, query table: tv_screener_stocks. Fields that have timeframe breakdown get a |# added, for example, Change percent for 60 minutes is change|60 /n
common fields used: Symbol Type:type,  earnings_per_share_basic_ttm, Change %:change (has timefremes), market_cap_basic, price_earnings_ttm, sector, Technical Rating: Recommend.All (has timeframe), volume (has timeframe), Volume*Price: Value.Traded (has timeframe), 	1-Month High: High.1M (extended format to Low, Perf (performance) and 3M etc.), All Time High: High.All (extendible to Low and Perf), 	Basic EPS (FY): basic_eps_net_income, cash_n_equivalents_fy, ebitda_yoy_growth_fy, earnings_per_share_diluted_yoy_growth_ttm, gross_margin, gross_profit_yoy_growth_ttm, Moving Averages Rating: Recommend.MA (has timeframes), Net Income (Quarterly YoY Growth): net_income_yoy_growth_fq, and so on. /n
you always want to select fields name, and close minimum for all queries. /n
"""

##Base Classes and Enums

tools = [
    list_database_tables,
    get_table_details,
    get_primary_keys,
    get_indexes,
    get_sample_rows,
    execute_raw_sql_query,
    list_orm_tables,
    get_orm_model_details,
    execute_orm_query,
    query_wolfram_alpha,
    get_market_news,
    get_fixed_window_analytics,
    get_sliding_window_analytics,
    get_stock_quote,
    repl_tool,
    update_trading_chart,
    get_stock_overview,
    run_tradingview_scan,
]

graph = create_react_agent(model, tools=tools, state_modifier=prompt, checkpointer=memory)

def print_stream(stream):
    for s in stream:
        if "messages" in s:
            # Access the last message in the list of messages
            message = s["messages"][-1]
            if isinstance(message, dict) and "content" in message:
                # Print the content of the message
                print(message["content"])
            else:
                print(message)
        else:
            print("Stream output does not contain 'messages':", s)