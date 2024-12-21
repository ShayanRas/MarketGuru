import os
from sqlalchemy import inspect
from langgraph.prebuilt import create_react_agent, ToolNode
from sqlalchemy import create_engine, MetaData
from sqlalchemy.engine.reflection import Inspector
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from sqlalchemy.orm import sessionmaker
from typing import Optional, Iterator, Any
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
from contextlib import contextmanager
from sqlalchemy.pool import QueuePool
from langgraph.checkpoint.postgres import PostgresSaver
from psycopg2 import connect
from sqlalchemy.engine.base import Engine
import psycopg2

DATABASE_URL = os.getenv("DATABASE_URL")
class SQLAlchemyConnectionPool:
    def __init__(self, engine):
        self.engine = engine
        
    @contextmanager
    def connection(self) -> Iterator[Any]:
        """Get a connection from the pool."""
        with self.engine.connect() as conn:
            yield conn

# Create the engine and connect
engine = create_engine(
    DATABASE_URL,
    poolclass=QueuePool,
    pool_size=20,
    max_overflow=0,
    pool_timeout=30,
    pool_pre_ping=True
)


def get_psycopg2_connection():
    return psycopg2.connect(
        dbname="marketguru_data",
        user="marketguru_data_user",
        password="2PRePLr1mLxqJtehIHHfUZCmWCzh0Zv6",
        host="dpg-ctif4vbtq21c73dnuc60-a.oregon-postgres.render.com",
        port=5432
    )

with get_psycopg2_connection() as conn:
    checkpointer = PostgresSaver(conn)
    checkpointer.setup()
    conn.commit()



metadata = MetaData()
metadata.reflect(bind=engine)


# Create an Inspector instance
inspector = inspect(engine)

Session = sessionmaker(bind=engine)
session = Session()

#model = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0.1)
model = ChatOpenAI(model="gpt-4o", temperature=0.1)

prompt = "You are a full CFA certified Trading and Investing focused AI assistant and chatbot. You have access to several tools that help you do this."

##Base Classes and Enums

##ORM pre definition


Base = declarative_base()

class EconBaseModel(BaseModel):
    interval: Optional[str] = "monthly"
    datatype: Optional[str] = "json"
class IncomeStatement(Base):
    __tablename__ = "income_statement"

    symbol = Column(String(10), primary_key=True)
    fiscaldateending = Column(Date, primary_key=True)
    reportedcurrency = Column(String(10))
    grossprofit = Column(BigInteger)
    totalrevenue = Column(BigInteger)
    costofrevenue = Column(BigInteger)
    costofgoodsandservicessold = Column(BigInteger)
    operatingincome = Column(BigInteger)
    sellinggeneralandadministrative = Column(BigInteger)
    researchanddevelopment = Column(BigInteger)
    operatingexpenses = Column(BigInteger)
    investmentincomenet = Column(String(50))
    netinterestincome = Column(BigInteger)
    interestincome = Column(String(50))
    interestexpense = Column(BigInteger)
    noninterestincome = Column(BigInteger)
    othernonoperatingincome = Column(BigInteger)
    depreciation = Column(BigInteger)
    depreciationandamortization = Column(BigInteger)
    incomebeforetax = Column(BigInteger)
    incometaxexpense = Column(BigInteger)
    interestanddebtexpense = Column(BigInteger)
    netincomefromcontinuingoperations = Column(BigInteger)
    comprehensiveincomenetoftax = Column(BigInteger)
    ebit = Column(BigInteger)
    ebitda = Column(BigInteger)
    netincome = Column(BigInteger)


class BalanceSheet(Base):
    __tablename__ = "balance_sheet"

    symbol = Column(String(10), primary_key=True)
    fiscaldateending = Column(Date, primary_key=True)
    reportedcurrency = Column(String(10))
    totalassets = Column(BigInteger)
    totalcurrentassets = Column(BigInteger)
    cashandcashequivalentsatcarryingvalue = Column(BigInteger)
    cashandshortterminvestments = Column(BigInteger)
    inventory = Column(BigInteger)
    currentnetreceivables = Column(BigInteger)
    totalnoncurrentassets = Column(BigInteger)
    propertyplantequipment = Column(BigInteger)
    accumulateddepreciationamortizationppe = Column(String(50))
    intangibleassets = Column(BigInteger)
    intangibleassetsexcludinggoodwill = Column(BigInteger)
    goodwill = Column(BigInteger)
    investments = Column(BigInteger)
    longterminvestments = Column(BigInteger)
    shortterminvestments = Column(BigInteger)
    othercurrentassets = Column(String(50))
    othernoncurrentassets = Column(String(50))
    totalliabilities = Column(BigInteger)
    totalcurrentliabilities = Column(BigInteger)
    currentaccountspayable = Column(BigInteger)
    deferredrevenue = Column(BigInteger)
    currentdebt = Column(BigInteger)
    shorttermdebt = Column(BigInteger)
    totalnoncurrentliabilities = Column(BigInteger)
    capitalleaseobligations = Column(BigInteger)
    longtermdebt = Column(BigInteger)
    currentlongtermdebt = Column(BigInteger)
    longtermdebtnoncurrent = Column(BigInteger)
    shortlongtermdebttotal = Column(BigInteger)
    othercurrentliabilities = Column(BigInteger)
    othernoncurrentliabilities = Column(BigInteger)
    totalshareholderequity = Column(BigInteger)
    treasurystock = Column(BigInteger)
    retainedearnings = Column(BigInteger)
    commonstock = Column(BigInteger)
    commonstocksharesoutstanding = Column(BigInteger)


class CashFlows(Base):
    __tablename__ = "cash_flows"

    symbol = Column(String(10), primary_key=True)
    fiscaldateending = Column(Date, primary_key=True)
    reportedcurrency = Column(String(10))
    operatingcashflow = Column(BigInteger)
    paymentsforoperatingactivities = Column(BigInteger)
    proceedsfromoperatingactivities = Column(String(50))
    changeinoperatingliabilities = Column(BigInteger)
    changeinoperatingassets = Column(BigInteger)
    depreciationdepletionandamortization = Column(BigInteger)
    capitalexpenditures = Column(BigInteger)
    changeinreceivables = Column(BigInteger)
    changeininventory = Column(BigInteger)
    profitloss = Column(BigInteger)
    cashflowfrominvestment = Column(BigInteger)
    cashflowfromfinancing = Column(BigInteger)
    proceedsfromrepaymentsofshorttermdebt = Column(BigInteger)
    paymentsforrepurchaseofcommonstock = Column(String(50))
    paymentsforrepurchaseofequity = Column(String(50))
    paymentsforrepurchaseofpreferredstock = Column(String(50))
    dividendpayout = Column(BigInteger)
    dividendpayoutcommonstock = Column(BigInteger)
    dividendpayoutpreferredstock = Column(String(50))
    proceedsfromissuanceofcommonstock = Column(String(50))
    proceedsfromissuanceoflongtermdebtandcapitalsecuritiesnet = Column(BigInteger)
    proceedsfromissuanceofpreferredstock = Column(String(50))
    proceedsfromrepurchaseofequity = Column(BigInteger)
    proceedsfromsaleoftreasurystock = Column(String(50))
    changeincashandcashequivalents = Column(String(50))
    changeinexchangerate = Column(String(50))
    netincome = Column(BigInteger)


class NewsSentimentInput(BaseModel):
    tickers: Optional[str] = None
    topics: Optional[str] = None
    time_from: Optional[str] = None
    time_to: Optional[str] = None
    sort: Optional[str] = "LATEST"
    limit: Optional[int] = 50


class AnalyticsInterval(str, Enum):
    ONE_MIN = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    THIRTY_MIN = "30min"
    SIXTY_MIN = "60min"
    DAILY = "DAILY"
    WEEKLY = "WEEKLY"
    MONTHLY = "MONTHLY"

class AnalyticsCalculation(str, Enum):
    MIN = "MIN"
    MAX = "MAX"
    MEAN = "MEAN"
    MEDIAN = "MEDIAN"
    CUMULATIVE_RETURN = "CUMULATIVE_RETURN"
    VARIANCE = "VARIANCE"
    STDDEV = "STDDEV"
    MAX_DRAWDOWN = "MAX_DRAWDOWN"
    HISTOGRAM = "HISTOGRAM"
    AUTOCORRELATION = "AUTOCORRELATION"
    COVARIANCE = "COVARIANCE"
    CORRELATION = "CORRELATION"

class DataType(str, Enum):
    JSON = "json"
    CSV = "csv"



def query_wolfram_alpha(input_query: str, maxchars: int = 6800) -> dict:
    """
    Query the Wolfram|Alpha LLM API for computational answers.

    Args:
        input_query: The natural language query to send to Wolfram|Alpha.
        maxchars: Maximum character limit for the response (default: 6800).

    Returns:
        A dictionary containing the response from Wolfram|Alpha or an error message.
    """
    # Retrieve the AppID from environment variables
    wolfram_appid = os.getenv("WOLFRAM_APPID")
    if not wolfram_appid:
        raise RuntimeError("Wolfram|Alpha AppID is not set in the environment variables.")

    # Base API URL
    base_url = "https://www.wolframalpha.com/api/v1/llm-api"

    # Construct the full URL with query parameters
    query_params = {
        "input": input_query,
        "appid": wolfram_appid,
        "maxchars": maxchars,
    }
    url = f"{base_url}?{urlencode(query_params)}"

    try:
        # Make the API request
        response = requests.get(url)

        # Raise an error if the status code is not 200
        response.raise_for_status()

        # Attempt to parse the response as JSON
        try:
            return response.json()
        except ValueError:
            return {
                "error": "Unexpected response format (not JSON).",
                "raw_response": response.text,
            }
    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to query Wolfram|Alpha: {str(e)}"}

def list_database_tables() -> list:
    """
    Retrieve all table names from the database.
    Use this tool to get a list of available tables in the SQL database.

    Returns:
        A list of table names.
    """
    global metadata, engine

    try:
        # Reflect metadata (ensures it's up to date)
        metadata.reflect(bind=engine)

        # Fetch table names
        tables = list(metadata.tables.keys())
        return tables
    except Exception as e:
        raise RuntimeError(f"Failed to list tables: {e}")


def get_table_details(table_name: str) -> dict:
    """
    Retrieve details of a specific table, including its columns, types, and nullable status.

    Args:
        table_name: Name of the table to fetch details for.

    Returns:
        A dictionary containing column details for the specified table.
    """
    global metadata

    if not table_name:
        raise ValueError("Table name must be provided.")

    try:
        # Check if the table exists
        if table_name not in metadata.tables:
            raise ValueError(f"Table '{table_name}' does not exist in the database.")

        # Get table columns and details
        table = metadata.tables[table_name]
        columns = [
            {
                "name": column.name,
                "type": str(column.type),
                "nullable": column.nullable,
            }
            for column in table.columns
        ]

        return {
            "table_name": table_name,
            "columns": columns,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get table details for '{table_name}': {e}")


def get_primary_keys(table_name: str) -> dict:
    """
    Retrieve the primary key(s) for a specified table.

    Args:
        table_name: Name of the table to fetch primary keys for.

    Returns:
        A dictionary containing the primary keys for the table.
    """
    global inspector

    if not table_name:
        raise ValueError("Table name must be provided.")

    try:
        # Fetch primary key constraints
        primary_keys = inspector.get_pk_constraint(table_name)

        if not primary_keys or not primary_keys.get("constrained_columns"):
            raise ValueError(f"No primary keys found for table '{table_name}'.")

        return {
            "table_name": table_name,
            "primary_keys": primary_keys.get("constrained_columns", []),
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get primary keys for '{table_name}': {e}")


def get_indexes(table_name: str) -> dict:
    """
    Retrieve the indexes for a specified table.

    Args:
        table_name: Name of the table to fetch indexes for.

    Returns:
        A dictionary containing index details for the specified table.
    """
    global inspector

    if not table_name:
        raise ValueError("Table name must be provided.")

    try:
        # Fetch indexes for the table
        indexes = inspector.get_indexes(table_name)

        if not indexes:
            return {
                "table_name": table_name,
                "indexes": [],
            }

        return {
            "table_name": table_name,
            "indexes": indexes,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to get indexes for '{table_name}': {e}")



def get_sample_rows(table_name: str, limit: int = 5) -> dict:
    """
    Retrieve a sample of rows from a specified table.

    Args:
        table_name: Name of the table to fetch rows from.
        limit: Number of rows to fetch (default: 5).

    Returns:
        A dictionary containing sample rows for the specified table.
    """
    global metadata, engine

    if not table_name:
        raise ValueError("Table name must be provided.")

    if limit <= 0:
        raise ValueError("Limit must be greater than 0.")

    try:
        # Check if the table exists
        if table_name not in metadata.tables:
            raise ValueError(f"Table '{table_name}' does not exist in the database.")

        # Retrieve the table object
        table = metadata.tables[table_name]

        # Execute query to get sample rows
        with engine.connect() as connection:
            result = connection.execute(table.select().limit(limit))
            rows = result.fetchall()

        # Format rows into a readable structure
        formatted_rows = [dict(zip(result.keys(), row)) for row in rows]

        return {
            "table_name": table_name,
            "sample_rows": formatted_rows,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to fetch sample rows for '{table_name}': {e}")


def execute_raw_sql_query(query: str) -> dict:
    """
    Execute a raw SQL query and return the results.

    Args:
        query: The SQL query to execute.

    Returns:
        A dictionary containing the results of the query, or an error message if execution fails.
    """
    global engine

    if not query:
        raise ValueError("Query must be provided.")

    try:
        # Execute the query
        with engine.connect() as connection:
            result = connection.execute(text(query))
            rows = result.fetchall()

        # Format rows into a readable structure
        if rows:
            column_names = result.keys()  # Get column names from the result
            formatted_rows = [dict(zip(column_names, row)) for row in rows]
        else:
            formatted_rows = []

        return {
            "query": query,
            "results": formatted_rows,
        }
    except Exception as e:
        raise RuntimeError(f"Failed to execute query '{query}': {e}")


def list_orm_tables() -> dict:
    """
    List all tables available in the ORM.

    Returns:
        A dictionary containing the table names and their corresponding model names.
    """
    global Base

    try:
        tables = {
            mapper.class_.__tablename__: mapper.class_.__name__
            for mapper in Base.registry.mappers
        }

        return {"tables": tables}
    except Exception as e:
        raise RuntimeError(f"Failed to list ORM tables: {e}")



def get_orm_model_details(table_name: str) -> dict:
    """
    Retrieve details of a specific ORM model, including its columns, types, and nullability.

    Args:
        table_name: Name of the ORM table to fetch details for.

    Returns:
        A dictionary containing model details (e.g., columns, types, nullability).
    """
    global Base

    if not table_name:
        raise ValueError("Table name must be provided.")

    try:
        # Find the model by its table name
        orm_model = next(
            mapper.class_
            for mapper in Base.registry.mappers
            if mapper.class_.__tablename__ == table_name
        )

        # Get column details
        columns = [
            {
                "name": column.name,
                "type": str(column.type),
                "nullable": column.nullable,
                "primary_key": column.primary_key,
            }
            for column in orm_model.__table__.columns
        ]

        return {
            "table_name": table_name,
            "columns": columns,
        }
    except StopIteration:
        raise ValueError(f"Table '{table_name}' not found in the ORM.")
    except Exception as e:
        raise RuntimeError(f"Failed to get model details for '{table_name}': {e}")


from sqlalchemy.orm import Session
from langchain_core.tools import tool


def execute_orm_query(model_name: str, filters: dict = None, limit: int = None) -> dict:
    """
    Execute an ORM-based query on a specified model.

    Args:
        model_name: The name of the ORM model (e.g., "IncomeStatement").
        filters: A dictionary of filters to apply (e.g., {"symbol": "IBM"}).
        limit: The maximum number of rows to return (optional).

    Returns:
        A dictionary containing the query results, or an error message if execution fails.
    """
    global session, Base

    if not model_name:
        raise ValueError("Model name must be provided.")

    try:
        # Dynamically retrieve the ORM model class using Base.registry.mappers
        orm_model = next(
            mapper.class_
            for mapper in Base.registry.mappers
            if mapper.class_.__name__ == model_name
        )

        # Start building the query
        query = session.query(orm_model)

        # Apply filters if provided
        if filters:
            query = query.filter_by(**filters)

        # Apply limit if provided
        if limit:
            query = query.limit(limit)

        # Execute the query and fetch results
        results = query.all()

        # Format results into dictionaries
        formatted_results = [
            {column.name: getattr(row, column.name) for column in orm_model.__table__.columns}
            for row in results
        ]

        return {
            "model_name": model_name,
            "filters": filters,
            "limit": limit,
            "results": formatted_results,
        }
    except StopIteration:
        raise ValueError(f"Model '{model_name}' not found in the ORM registry.")
    except Exception as e:
        raise RuntimeError(f"Failed to execute ORM query on '{model_name}': {e}")

import os
import requests
from langchain_core.tools import tool

import os
import requests
from langchain_core.tools import tool
from urllib.parse import urlencode


def parse_plaintext_response(raw_response: str, input_query: str) -> dict:
    """
    Parse non-JSON responses from Wolfram|Alpha and return a structured output.

    Args:
        raw_response: The raw text response from the Wolfram|Alpha API.
        input_query: The original query sent to Wolfram|Alpha.

    Returns:
        A structured dictionary representation of the raw response.
    """
    # Extract meaningful parts from the raw response
    lines = raw_response.strip().split("\n")
    structured_output = {"query": input_query, "response_lines": []}

    for line in lines:
        if line.strip():
            structured_output["response_lines"].append(line.strip())

    return structured_output

import os
import requests
from langchain_core.tools import tool
from urllib.parse import urlencode

@tool
def query_wolfram_alpha(input_query: str = None, maxchars: int = 6800) -> str:
    """
    Query the Wolfram|Alpha LLM API and return a clean text response.

    Args:
        input_query: The natural language query to send to Wolfram|Alpha.
        maxchars: Maximum character limit for the response (default: 6800).

    Returns:
        A clean string response from Wolfram|Alpha.
    """
    # Retrieve the AppID from environment variables
    wolfram_appid = os.getenv("WOLFRAM_ALPHA_APPID")
    if not wolfram_appid:
        return "Error: Wolfram|Alpha AppID is not set in the environment variables."

    # Ensure input_query is provided
    if not input_query:
        return "Error: No query provided for Wolfram|Alpha."

    # Base API URL
    base_url = "https://www.wolframalpha.com/api/v1/llm-api"

    # Construct the API request URL
    query_params = {
        "input": input_query,
        "appid": wolfram_appid,
        "maxchars": maxchars,
    }
    url = f"{base_url}?{urlencode(query_params)}"

    try:
        # Log the constructed URL for debugging
        print(f"Constructed API URL: {url}")

        # Make the API request
        response = requests.get(url)

        # Log the raw response
        print(f"Response Status Code: {response.status_code}")
        print(f"Response Headers: {response.headers}")
        print(f"Raw Response Text: {response.text}")

        # Raise an error if the response status is not 200
        response.raise_for_status()

        # Check response format
        if "application/json" in response.headers.get("Content-Type", ""):
            # If JSON, parse and return as clean text
            json_data = response.json()
            return json_data.get("result", "No result provided.")
        else:
            # If plain text, return as is
            return response.text.strip()

    except requests.exceptions.RequestException as e:
        return f"Error: Failed to query Wolfram|Alpha - {str(e)}"


@tool
def get_market_news(
    tickers: Optional[str] = None,
    topics: Optional[str] = None, 
    time_from: Optional[str] = None,
    time_to: Optional[str] = None,
    sort: Optional[str] = "LATEST",
    limit: Optional[int] = 50
) -> dict:
    """Get market news and sentiment data from Alpha Vantage.
    
    Args:
        tickers: Optional stock/crypto/forex symbols (e.g. 'AAPL' or 'COIN,CRYPTO:BTC')
        topics: Optional news topics (topics: blockchain, earnings, ipo, mergers_and_acquisitions, financial_markets, economy_fiscal, economy_monetary, economy_macro, energy_transportation, finance, life_sciences, manufacturing, real_estate, retail_wholesale, technology) 
        time_from: Optional start time in YYYYMMDDTHHMM format
        time_to: Optional end time in YYYYMMDDTHHMM format
        sort: Optional sorting ('LATEST', 'EARLIEST', or 'RELEVANCE')
        limit: Optional limit of results (default 50, max 1000)
    """
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'NEWS_SENTIMENT',
        'apikey': os.getenv("ALPHAVANTAGE_API_KEY")
    }
    
    # Add optional parameters if provided
    if tickers: params['tickers'] = tickers
    if topics: params['topics'] = topics
    if time_from: params['time_from'] = time_from
    if time_to: params['time_to'] = time_to
    if sort: params['sort'] = sort
    if limit: params['limit'] = limit
    
    try:
        response = requests.get(url, params=params)
        data = response.json()
        return data
    except Exception as e:
        return {"error": str(e)}
@tool
def get_fixed_window_analytics(
    symbols: str,
    range_start: str,
    range_end: Optional[str] = None,
    interval: str = "DAILY",
    ohlc: str = "close",
    calculations: str = "MEAN,STDDEV,CORRELATION"
) -> dict:
    """Calculate fixed-window analytics for given symbols.
    
    Args:
        symbols: Comma-separated symbols (e.g., 'AAPL,MSFT')
        range_start: Start date (YYYY-MM-DD) or period (e.g., '5day')
        range_end: Optional end date (YYYY-MM-DD)
        interval: Time interval (e.g., '1min', '5min', 'DAILY')
        ohlc: Price type ('open', 'high', 'low', 'close')
        calculations: Metrics to calculate (e.g., 'MEAN,STDDEV,CORRELATION')
    """
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'ANALYTICS_FIXED_WINDOW',
        'SYMBOLS': symbols,
        'INTERVAL': interval,
        'OHLC': ohlc,
        'CALCULATIONS': calculations,
        'apikey': os.getenv("ALPHAVANTAGE_API_KEY")
    }
    # Construct RANGE parameter correctly
    if range_end:
        params['RANGE'] = f"{range_start},{range_end}"  # Use comma or appropriate delimiter
    else:
        params['RANGE'] = range_start
    return requests.get(url, params=params).json()
@tool
def get_sliding_window_analytics(
    symbols: str,
    range_value: str,
    window_size: int,
    interval: str = "DAILY",
    ohlc: str = "close",
    calculations: str = "MEAN,STDDEV"
) -> dict:
    """Calculate sliding-window analytics for given symbols.
    
    Args:
        symbols: Comma-separated symbols (e.g. 'AAPL,MSFT')  
        range_value: Time range or YYYY-MM-DD
        window_size: Size of sliding window (min 10)
        interval: Time interval (1min to MONTHLY)
        ohlc: Price type (open/high/low/close)
        calculations: Metrics to calculate (MEAN,STDDEV,etc)
        
    Calculations: MEAN,MEDIAN,CUMULATIVE_RETURN,VARIANCE,STDDEV,
    COVARIANCE,CORRELATION(PEARSON/KENDALL/SPEARMAN)
    """
    url = 'https://www.alphavantage.co/query'
    params = {
        'function': 'ANALYTICS_SLIDING_WINDOW',
        'SYMBOLS': symbols,
        'RANGE': range_value,
        'WINDOW_SIZE': window_size,
        'INTERVAL': interval,
        'OHLC': ohlc,
        'CALCULATIONS': calculations,
        'apikey': os.getenv("ALPHAVANTAGE_API_KEY")
    }
    return requests.get(url, params=params).json()

import os
import requests
from langchain_core.tools import tool

@tool
def get_stock_quote(symbol: str, datatype: str = "json") -> dict:
    """
    Fetch the latest price and volume information for a specific ticker.

    Args:
        symbol: The stock ticker symbol to query (e.g., 'IBM').
        datatype: The response format, either 'json' or 'csv' (default: 'json').

    Returns:
        A dictionary containing the latest stock quote data or an error message.
    """
    # Ensure the API key is set in the environment variables
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "Alpha Vantage API key is not set in the environment variables."}

    # Define the base URL and parameters
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "GLOBAL_QUOTE",
        "symbol": symbol,
        "datatype": datatype,
        "apikey": api_key
    }

    try:
        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()

        # Handle JSON response
        if datatype == "json":
            data = response.json()
            if "Global Quote" in data:
                return data["Global Quote"]
            else:
                return {"error": "No data found for the specified symbol."}

        # Handle CSV response
        elif datatype == "csv":
            return {"data": response.text}

        else:
            return {"error": "Invalid datatype. Supported values are 'json' and 'csv'."}

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch stock quote: {str(e)}"}


# Create an instance of PythonREPL
python_repl = PythonREPL()

# Create the PythonREPL tool
repl_tool = Tool(
    name="python_repl",
    description="A Python shell. Use this to execute python commands. Input should be a valid python command. If you want to see the output of a value, you should print it out with `print(...)`.",
    func=python_repl.run,
)

@tool
def update_trading_chart(
    symbol: str,
    interval: str = "D",
    studies: list = None
) -> dict:
    """Update the trading chart display.
    
    Args:
        symbol: Trading symbol (e.g., 'NASDAQ:AAPL')
        interval: Time interval (e.g., 'D' for daily)
        studies: List of technical indicators
    """
    url = 'https://duxokbotzqdtmdbhgcns.functions.supabase.co/update-chart'
    payload = {
        'symbol': symbol,
        'interval': interval,
        'studies': studies or []
    }
    return requests.post(url, json=payload).json()

@tool
def get_stock_overview(symbol: str) -> dict:
    """
    get overview of quarterly stock fundamentals including CIK,Exchange,Currency,Country,Sector,Industry,Address,OfficialSite,FiscalYearEnd,LatestQuarter,MarketCapitalization,EBITDA,PERatio,PEGRatio,BookValue,DividendPerShare,DividendYield,EPS,RevenuePerShareTTM,ProfitMargin,OperatingMarginTTM,ReturnOnAssetsTTM,ReturnOnEquityTTM,RevenueTTM,GrossProfitTTM,DilutedEPSTTM,QuarterlyEarningsGrowthYOY,QuarterlyRevenueGrowthYOY,AnalystTargetPrice,AnalystRatingStrongBuy,AnalystRatingBuy,AnalystRatingHold,AnalystRatingSell,AnalystRatingStrongSell,TrailingPE,ForwardPE,PriceToSalesRatioTTM,PriceToBookRatio,EVToRevenue,EVToEBITDA,Beta,52WeekHigh,52WeekLow,50DayMovingAverage,200DayMovingAverage,SharesOutstanding,DividendDate,ExDividendDate

    Args:
        symbol: The stock ticker symbol to query (e.g., 'AAPL').

    Returns:
        A dictionary containing the latest stock overiview info data or an error message.
    """
    # Ensure the API key is set in the environment variables
    api_key = os.getenv("ALPHAVANTAGE_API_KEY")
    if not api_key:
        return {"error": "Alpha Vantage API key is not set in the environment variables."}

    # Define the base URL and parameters
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "OVERVIEW",
        "symbol": symbol,
        "apikey": api_key
    }

    try:
        # Make the API request
        response = requests.get(url, params=params)
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        return data

    except requests.exceptions.RequestException as e:
        return {"error": f"Failed to fetch stock overview: {str(e)}"}

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
    get_stock_overview
]

graph = create_react_agent(
    llm=model,
    tools=tools,
    state_modifier=prompt,
    checkpointer=checkpointer,
    debug=True
)

def run_agent_with_persistence(input_data: dict, thread_id: str) -> Iterator:
    """Run agent with persistence and streaming enabled"""
    config = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": ""
        }
    }
    
    # Get existing checkpoint if available
    existing_checkpoint = checkpointer.get(config)
    if existing_checkpoint:
        config["configurable"]["checkpoint_id"] = existing_checkpoint.id
        
    return graph.stream(
        input_data,
        config=config,
        stream_mode="values"
    )

def print_stream(stream: Iterator) -> None:
    """Process and print streaming output with proper formatting"""
    try:
        for s in stream:
            if "messages" in s:
                message = s["messages"][-1]
                
                # Handle tuple messages (human inputs)
                if isinstance(message, tuple):
                    print(f"Human: {message[1]}")
                    continue
                    
                # Handle AI messages with tool calls
                if hasattr(message, "tool_calls") and message.tool_calls:
                    print("\nAI Tool Calls:")
                    for tool_call in message.tool_calls:
                        print(f"- {tool_call.name}: {tool_call.args}")
                    continue
                
                # Handle tool messages
                if hasattr(message, "tool_name"):
                    print(f"\nTool Response ({message.tool_name}):")
                    print(message.content)
                    continue
                
                # Handle regular AI messages
                if hasattr(message, "content"):
                    print(f"\nAI: {message.content}")
                    continue
                
                # Fallback for other message types
                print(f"\nMessage: {message}")
            else:
                print(f"Other output: {s}")
                
    except Exception as e:
        print(f"Error processing stream: {str(e)}")