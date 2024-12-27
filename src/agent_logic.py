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
    execute_tradingview_query
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

prompt = """You are a full CFA certified Trading and Investing focused AI assistant and chatbot. You have access to several tools. Your job is to help answer questions about the market, trading, and investing-related queries. /n
Use a combination of your toolset as needed to answer the user's questions. /n
Prioritize the use of the `execute_tradingview_query` tool to query stock data by writing full TradingView Query objects directly. Fallback to the `get_stock_overview` tool for general stock information, and use the databases available for financials such as balance sheet, cash flows, and earnings. /n
Perform calculations via Wolfram Alpha if needed. If not possible, fallback to the Python REPL. /n
Refuse to perform actions unrelated to trading or investing. /n

Further instructions about tool usage for complex queries: /n
# Tool Guide: TradingView Screener (execute_tradingview_query) /n
Purpose: Allows you to construct and execute raw TradingView Screener queries directly in Python. This tool accepts full TradingView Query objects written as Python strings. /n
Stock Identifier: Use `name` or `ticker` to filter specific companies. **Symbol is not supported**. Example: "Apple Inc." → `Column('name') == 'AAPL'` or `Column('ticker') == 'NASDAQ:AAPL'`. /n
Essential Fields (Always Include): `name` (company name), `close` (latest closing price). /n
Common Fields: `High.1M` (1-month high), `High.All` (all-time high), `earnings_per_share_diluted_ttm` (EPS ttm), `earnings_per_share_diluted_yoy_growth_ttm` (YoY EPS growth), `market_cap_basic` (market cap), `sector` (industry), `volume` (timeframe-based), `price_earnings_ttm` (P/E ratio ttm), `dividends_yield_current` (dividend yield), `Recommend.All` (technical rating), `Value.Traded` (Volume * Price). /n
Timeframe Fields: Use |# for timeframe variations. Example: `change|60` (60-minute change), `High.3M` (3-month high), `Perf.1M` (1-month performance), `Recommend.MA|1D` (Moving Average daily). /n
Sectors for Filtering: Finance, Commercial Services, Process Industries, Communications, Consumer Durables, Consumer Non-Durables, Transportation, Health Technology, Retail Trade, Consumer Services, Technology Services, Electronic Technology, Miscellaneous, Distribution Services, Producer Manufacturing, Non-Energy Minerals, Health Services, Energy Minerals, Industrial Services, Utilities. /n

Examples (Full Query Objects): /n
1. "Top 10 companies by market cap": /n
(Query()
 .select('name', 'close', 'market_cap_basic')
 .order_by('market_cap_basic', ascending=False)
 .limit(10)) /n
2. "Tech sector stocks above $50": /n
(Query()
 .select('name', 'close', 'sector')
 .where(
     Column('sector') == 'Technology Services',
     Column('close') > 50
 )
 .order_by('name', ascending=True)
 .limit(50)) /n
3. "Apple's 1-month high, all-time high, and EPS growth": /n
(Query()
 .select('name', 'High.1M', 'High.All', 'earnings_per_share_diluted_yoy_growth_ttm')
 .where(
     Column('name') == 'AAPL'
 )
 .limit(1)) /n
4. "Large-cap financial stocks over 10 billion": /n
(Query()
 .select('name', 'close', 'sector', 'market_cap_basic')
 .where(
     Column('sector') == 'Finance',
     Column('market_cap_basic') > 10000000000
 )
 .order_by('market_cap_basic', ascending=False)
 .limit(20)) /n
5. "Energy stocks with the highest dividend yield": /n
(Query()
 .select('name', 'close', 'sector', 'dividends_yield_current')
 .where(
     Column('sector') == 'Energy Minerals'
 )
 .order_by('dividends_yield_current', ascending=False)
 .limit(25)) /n

Handling New Queries: When users request data or filters not explicitly covered in the examples, construct full TradingView Query objects by referencing `tv_screener_stocks`. If the user requests "best-performing" stocks, map this to performance-related fields like `Perf.1M` or `change|60`. If no direct field is found, prioritize relevant financial metrics like EPS, market cap, or sector performance. Always include `name` and `close` in results, even if not explicitly mentioned. /n
For specific stock queries, use `ticker` (e.g., `NASDAQ:SMX`, `OTC:CNSWF`) or `name` (e.g., `SMX`, `CNSWF`). /n
###end guide for execute_tradingview_query tool /n

When the user asks about a stock, after retrieving the relevant info, call the `update_trading_chart` tool to update the chart for the stock name (ticker). /n
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
    execute_tradingview_query,
    get_time_series_daily_adjusted
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