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
# Tool Guide: TradingView Screener (run_tradingview_scan) /n
Purpose: Queries TradingView's Screener API to dynamically filter and return stock data based on user-defined conditions. /n
Stock Identifier: Use symbol (not name) to filter specific companies. Example: "Apple Inc." → {"column": "symbol", "operation": "equal", "value": "AAPL"} /n
Essential Fields (Always Include): name (company name), close (latest closing price). /n
Common Fields: High.1M (1-month high), High.All (all-time high), earnings_per_share_diluted_ttm (EPS ttm), earnings_per_share_diluted_yoy_growth_ttm (YoY EPS growth), market_cap_basic (market cap), sector (industry), volume (timeframe-based), price_earnings_ttm (P/E ratio ttm), dividends_yield_current (dividend yield), Recommend.All (technical rating), Value.Traded (Volume * Price). /n
Timeframe Fields: Use |# for timeframe variations. Example: change|60 (60-minute change), High.3M (3-month high), Perf.1M (1-month performance), Recommend.MA|1D (Moving Average daily). /n
Sectors for Filtering: Finance, Commercial Services, Process Industries, Communications, Consumer Durables, Consumer Non-Durables, Transportation, Health Technology, Retail Trade, Consumer Services, Technology Services, Electronic Technology, Miscellaneous, Distribution Services, Producer Manufacturing, Non-Energy Minerals, Health Services, Energy Minerals, Industrial Services, Utilities. /n
Examples: /n
1. "Top 10 companies by market cap" → {"select_fields": ["name", "close", "market_cap_basic"], "filter_conditions": [], "order_by_field": "market_cap_basic", "ascending": False, "limit": 10} /n
2. "Tech sector stocks above $50" → {"select_fields": ["name", "close", "sector"], "filter_conditions": [{"column": "sector", "operation": "equal", "value": "Technology Services"}, {"column": "close", "operation": "greater", "value": 50}], "order_by_field": "name", "ascending": True, "limit": 50} /n
3. "Apple's 1-month high, all-time high, and EPS growth" → {"select_fields": ["name", "High.1M", "High.All", "earnings_per_share_diluted_yoy_growth_ttm"], "filter_conditions": [{"column": "symbol", "operation": "equal", "value": "AAPL"}], "order_by_field": "name", "ascending": True, "limit": 1} /n
4. "Large-cap financial stocks over 10 billion" → {"select_fields": ["name", "close", "sector", "market_cap_basic"], "filter_conditions": [{"column": "sector", "operation": "equal", "value": "Finance"}, {"column": "market_cap_basic", "operation": "greater", "value": 10000000000}], "order_by_field": "market_cap_basic", "ascending": False, "limit": 20} /n
5. "Energy stocks with the highest dividend yield" → {"select_fields": ["name", "close", "sector", "dividends_yield_current"], "filter_conditions": [{"column": "sector", "operation": "equal", "value": "Energy Minerals"}], "order_by_field": "dividends_yield_current", "ascending": False, "limit": 25} /n
Handling New Queries: When users request data or filters not explicitly covered in the examples, infer the closest matching fields by referencing tv_screener_stocks. If the user requests "best-performing" stocks, map this to performance-related fields like Perf.1M or change|60. If no direct field is found, prioritize relevant financial metrics like EPS, market cap, or sector performance. Always include name and close in results, even if not explicitly mentioned. Use broad filtering for exploratory queries and refine as the user provides more specific criteria. /n
to select for a specific stock, either use ticker (ticket format example: NASDAQ:SMX, OTC:CNSWF	etc) or name (example: SMX, CNSWF)

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