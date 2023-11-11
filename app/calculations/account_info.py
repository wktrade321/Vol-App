# %%
import numpy as np
import pandas as pd
import tda
from tda.auth import easy_client
from tda.client import Client
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
import httpx
import json
from datetime import date, datetime, timedelta
import lxml
import html5lib
from bs4 import BeautifulSoup
import requests
from IPython.display import display

from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
import selenium.common.exceptions


chrome_options = Options()
chrome_options.add_argument('--no-sandbox')
chrome_options.add_argument('--headless')
chrome_options.add_argument('--disable-dev-shm-usage')
chrome_options.add_argument('--start-maximized')
chrome_options.page_load_strategy = 'eager'


import os
from dotenv import load_dotenv

load_dotenv('e.env')


# %%
#initialize chromedriver function 
def driver():
    return webdriver.Chrome(service=Service(ChromeDriverManager().install()))

#initialize TDA easy_client
c = easy_client(
    webdriver_func=driver,
    api_key=os.environ['tda_api_key'],
    redirect_uri='https://localhost',
    token_path='token.json'
)


# %%
def get_trades(account_id='255715792', lookback: int=365, start_date=None, write_to_file: bool=True):    
    r = c.get_transactions(account_id=account_id, transaction_type=Client.Transactions.TransactionType('TRADE'))
    assert r.status_code == httpx.codes.OK, r.raise_for_status()
    txns = pd.read_json(r)[['netAmount', 'transactionDate']]
    trades = list(pd.read_json(r)['transactionItem'])
    trades = pd.DataFrame(trades)
    inst = pd.DataFrame(list(trades['instrument']))
    trades[['underlying', 'putCall']] = inst[['underlyingSymbol', 'putCall']]
    txns[['amount', 'price', 'cost', 'instruction', 'positionEffect', 'underlying', 'putCall']] = trades[['amount', 'price', 'cost', 'instruction', 'positionEffect', 'underlying', 'putCall']]
    txns.set_index('transactionDate', inplace=True)
    txns.index = pd.to_datetime(txns.index).tz_localize(None)
    if start_date is not None:
        trades = txns[txns.index >= start_date]
    else:   
        trades = txns[txns.index >= datetime.today()-timedelta(lookback)]

    trades['date'] = trades.index.date
    trades = trades.groupby(by=['underlying', 'positionEffect', 'date'],as_index=False)['cost'].sum()
    trades['cost'] = trades['cost']*-1
    open = trades[trades['positionEffect']=='OPENING'].set_index('underlying')
    clos = trades[trades['positionEffect']=='CLOSING'].set_index('underlying')

    trades2 = pd.merge(open, clos, how='outer', left_index=True, right_index=True, suffixes=['_open','_close'])
    trades3 = trades2[trades2['date_close'] > trades2['date_open']].reset_index().drop_duplicates(subset=['underlying','date_close'],keep='last')
    trades4 = trades3.groupby(['underlying','date_open'])['cost_close'].sum()
    trades5 = trades3.groupby(['underlying','date_open'])['date_close'].last()
    trades6 = pd.merge(open, trades4, how='left', left_on=['underlying','date'], right_index=True)
    trades7 = pd.merge(trades6, trades5, how='left', left_on=['underlying','date'], right_index=True) \
                        .drop('positionEffect',axis=1).rename(columns={'cost': 'cost_open', 'date':'date_open'}) \
                        .reindex(['date_open','cost_open','date_close','cost_close'],axis=1)
    trades7['cost_close'] = trades7['cost_close']*-1
    
    if write_to_file:
        if start_date is not None:
            trades7.to_csv(f'trades_from_{start_date.date().month}-{start_date.date().day}_{datetime.today().replace(microsecond=0)}.csv')
        else:
            trades7.to_csv(f'trades_last{lookback}_{datetime.today().replace(microsecond=0)}.csv')
    
    return trades7

# %%
def get_watchlists(account_id='255715792'):    
    p = c.get_watchlists_for_single_account(account_id='255715792')
    assert p.status_code == httpx.codes.OK, p.raise_for_status()
    wls = pd.read_json(p)
    wl_dict = {}
    for name in wls['name']:
        wl = wls[wls.name == name]
        wlsymbols = {str(name): pd.Series([d['symbol'] for d in list(pd.DataFrame(list(wl.loc[:,'watchlistItems'])[0])['instrument'])])}
        wl_dict.update(wlsymbols)
    
    watchlists = pd.DataFrame(wl_dict)
    return watchlists

# %%
def get_txns(account_id='255715792',txn_type='ALL'):
    """ 
    txn_type = 'ALL' or 'TRADE' or 'BUY_ONLY' or 'SELL_ONLY' or 'CASH_IN_OR_CASH_OUT' or 'CHECKING' or 'DIVIDEND' or 'INTEREST' or 'OTHER' or 'ADVISORY_FEES' 
    """
    t = c.get_transactions(account_id=account_id,transaction_type=Client.Transactions.TransactionType(txn_type))
    assert t.status_code == httpx.codes.OK, t.raise_for_status()
    txns = list(pd.read_json(t)['transactionItem'])
    txns = pd.DataFrame(txns)
    inst = pd.DataFrame(list(txns['instrument']))
    txns = txns.join(inst[['underlying','optionExp','putCall']].reset_index())
    txns.join()
    return txns, inst

# %%
def get_trades_raw(filename_html):
    HTMLFileToBeOpened = open(filename_html, "r")
    open_pos = pd.read_html(HTMLFileToBeOpened,match='Options')[0][['Symbol','Exp','Strike','Type','Qty']].dropna(subset='Symbol')
    HTMLFileToBeOpened = open(filename_html, "r")
    trades = pd.read_html(HTMLFileToBeOpened,match='Account Trade History')[0].reset_index()
    trades['open_pos'] = np.where(pd.merge(trades,open_pos,how='left',on=['Symbol','Exp','Strike','Type','Qty'],indicator='exists').exists == 'both',1,0)
    return trades

# %%
### Get PNL info for each strategy in paper trading account using HTML file exported
def get_trades_grouped(trades_raw: pd.DataFrame):

    trades = trades_raw[trades_raw['open_pos'] != 1]

    trades = trades.query('Spread.str.len() > 0 and  Spread != "STOCK" and Spread != "COVERED"')

    #put relevant columns to numeriec
    trades[['Qty','Price','Net Price']] = trades[['Qty','Price','Net Price']].apply(pd.to_numeric)

    #calculated credits (+ for opening credits, - for closing debits and vice versa)
    trades['Credits'] = trades['Qty']*trades['Net Price']*-100

    #set trade time to dt
    trades['Exec Time'] = pd.to_datetime(trades['Exec Time'])

    #order by symbol then trade time
    trades.sort_values(['Symbol','Exec Time'], inplace=True)
    #trades.reset_index(inplace=True)

    #set index

    #create trade ID for each symbol/expiration combo (i.e. each spread)
    trades['trade_id'] = trades.groupby(['Symbol','Exp']).ngroup() + 1

    
    trades = trades[trades.groupby('trade_id').trade_id.transform('count')>1]   
    

    #order trades (opening and closing) for each trade_id
    trades['rownum'] = trades.groupby('trade_id')['Exec Time'].rank(method='first')

    trades.drop(trades[(trades['rownum'] == 1) & (trades['Pos Effect'] == 'TO CLOSE')].index, inplace=True)

    #map spreads to strategy
    def strat_map(spread):
        if (spread == 'IRON CONDOR') or (spread == 'STRADDLE') or (spread == 'STRANGLE'):
            return 'EARNINGS'
        elif (spread == 'CALENDAR') or (spread == 'BUTTERFLY'):
            return 'FORWARD VOL'
        elif (spread == '~BUTTERFLY'):
            return 'SKEW'
        else:
            return 'UNKNOWN'


    #apply spread to strategy mapping
    trades['Strategy'] = trades['Spread'].apply(strat_map)

    #create temp df for opening trades and unknown trades, to fill strategies for unknowns
    temp = trades.copy()[(trades['rownum'] == 1) | (trades['Strategy'] == 'UNKNOWN')]

    #for opening legs that are unknown, set to FORWARD VOL
    #temp['Strategy'] = np.where((temp['Strategy'] == 'UNKNOWN') & (temp['rownum']==1), 'FORWARD VOL', temp['Strategy'])

    #replace UNKNOWN with NaN for forward fill
    temp['Strategy'].replace('UNKNOWN',np.nan,inplace=True)

    #forward fill unknowns (NaNs) from opening trades
    temp['Strategy'].fillna(method='ffill',inplace=True)

    #merge filled strategies and replace UNKNOWN in original df with them
    trades = trades.merge(temp[['Symbol','Exec Time','Strategy']],how='left',on=['Symbol','Exec Time'])
    trades['Strategy'] = np.where(trades['Strategy_x'] == 'UNKNOWN', trades['Strategy_y'], trades['Strategy_x'])
    trades.drop(['Strategy_x','Strategy_y'],axis=1,inplace=True)

    trades_final = trades.copy()

    #identify winners and loserse based on sum of credits - put 1/0 on opening trade to keep unique
    trades_final['winner'] = np.where((trades_final.groupby('trade_id')['Credits'].transform('sum') > 0) \
                                            & (trades_final['rownum'] == 1), 1, 0)
    trades_final['loser'] = np.where((trades_final.groupby('trade_id')['Credits'].transform('sum') <= 0) \
                                            & (trades_final['rownum'] == 1), 1, 0)

    #sum winners and losers to get total number of trades
    trades_final['num_trades'] = trades_final['winner'] + trades_final['loser'] 

    #get opening and closing credits separately for each trade to calculate ROC, etc
    trades_final['Opening Credits'] = np.where(trades_final['Pos Effect'] == 'TO OPEN', trades_final['Credits'], 0)
    trades_final['Closing Credits'] = np.where(trades_final['Pos Effect'] == 'TO CLOSE', trades_final['Credits'], 0)

    return trades_final


# %%
def get_trades_wide(trades_grouped: pd.DataFrame):
    #pivot table by trade
    trades_wide = pd.pivot_table(trades_grouped, values = ['winner','loser','Opening Credits','Closing Credits'], index = ['trade_id'], aggfunc = np.sum).reset_index()
    
    trades_wide['trade_id'] = np.where((trades_wide['Opening Credits'] == 0) | (trades_wide['Closing Credits'] == 0),trades_wide['trade_id'] + 1, trades_wide['trade_id'])

    #trades_wide = pd.pivot_table(trades_wide, values = ['winner','loser','Opening Credits','Closing Credits'], index = ['trade_id'], aggfunc = np.sum).reset_index()
    
    trades_wide = pd.merge(trades_wide,trades_grouped[['trade_id','Symbol','Strategy']].drop_duplicates(),how='inner',on='trade_id')

    

    #calculations for each trade
    trades_wide['Profit'] = trades_wide['Opening Credits'] + trades_wide['Closing Credits']
    trades_wide['roc'] = trades_wide['Profit']/np.abs(trades_wide['Opening Credits'])
    trades_wide['Winner Profit'] = trades_wide['winner']*trades_wide['Profit']
    trades_wide['Loser Profit'] = trades_wide['loser']*np.abs(trades_wide['Profit'])
    

    trades_wide = trades_wide.drop_duplicates(subset='trade_id')
    return trades_wide

# %%
def get_pnl(trades_wide):
    pnl = pd.pivot_table(trades_wide, values = ['Opening Credits','Closing Credits','winner','loser','Profit','Winner Profit','Loser Profit'],index='Strategy',aggfunc=np.sum)

    pnl['num_trades'] = pnl['loser'] + pnl['winner']
    pnl['Win Rate'] = pnl['winner']/pnl['num_trades']
    
    pnl['ROC'] = pnl['Profit']/np.abs(pnl['Opening Credits'])

    pnl['Avg Capital'] = np.abs(pnl['Opening Credits'])/pnl['num_trades']
    pnl['Avg Proft'] = pnl['Profit']/pnl['num_trades']

    pnl['Avg Winner'] = pnl['Winner Profit']/pnl['winner']
    pnl['Avg Loser'] = pnl['Loser Profit']/pnl['loser']

    pnl['Kelly %'] = pnl['Win Rate'] - (1-pnl['Win Rate'])/(pnl['Avg Winner']/pnl['Avg Loser'])

    return pnl

# %%
def get_trades_by_exp(account_id: str='255715792', lookback: int=365, start_date=None, write_to_file: bool=True, symbols: list=None):
    r = c.get_transactions(account_id=account_id, transaction_type=Client.Transactions.TransactionType('TRADE'))
    assert r.status_code == httpx.codes.OK, r.raise_for_status()
    txns = pd.read_json(r)[['netAmount', 'transactionDate']]
    trades = list(pd.read_json(r)['transactionItem'])
    trades = pd.DataFrame(trades)

    inst = pd.DataFrame(list(trades['instrument']))
    trades[['underlying', 'exp', 'putCall', 'cusip']] = inst[['underlyingSymbol', 'optionExpirationDate', 'putCall', 'cusip']]
    if symbols is not None:
        trades = trades[trades['underlying'].isin(symbols)]

    trades['strike'] = pd.to_numeric(trades['cusip'].str[-7:])/1000

    txns[['amount', 'price', 'cost', 'instruction', 'positionEffect', 'underlying', 'exp', 'strike', 'putCall']] = \
        trades[['amount', 'price', 'cost', 'instruction', 'positionEffect', 'underlying', 'exp', 'strike', 'putCall']]
    
    txns.set_index('transactionDate', inplace=True)
    txns.index = pd.to_datetime(txns.index).tz_localize(None)
    if start_date is not None:
        trades = txns[txns.index >= start_date]
    else:   
        trades = txns[txns.index >= datetime.today()-timedelta(lookback)]

    trades['date'] = trades.index.date
    trades['exp'] = pd.to_datetime(trades['exp']).dt.date


    closing = trades[trades['positionEffect'] == 'CLOSING']
    opening = trades[trades['positionEffect'] == 'OPENING']
    closing2 = closing.groupby(by=['underlying','exp','strike'])[['amount','cost']].sum()
    opening2 = opening.groupby(by=['underlying','exp','strike'])[['amount','cost']].sum()
    closing3 = closing.groupby(by=['underlying','exp','strike'])['date'].max()
    opening3 = opening.groupby(by=['underlying','exp','strike'])['date'].min()
    trades2 = pd.merge(opening2, closing2, how='outer', left_index=True, right_index=True, suffixes=['_open','_close'])
    trades2['cost_open'] = trades2['cost_open']*-1

    trades2 = pd.merge(trades2, opening3, how='left', left_index=True, right_index=True)
    trades2 = pd.merge(trades2, closing3, how='left', left_index=True, right_index=True, suffixes = ['_open','_close'])

    trades2['date_close'] = trades2['date_close'].where(trades2['amount_open']==trades2['amount_close'], np.nan)

    trades2 = trades2[['date_open', 'amount_open', 'cost_open', 'date_close', 'amount_close', 'cost_close']]
    if write_to_file:
        if start_date is not None:
            trades2.to_csv(f'trades_by_exp_from_{start_date.date().month}-{start_date.date().day}_{datetime.today().replace(microsecond=0)}.csv')
        else:
            trades2.to_csv(f'trades_by_exp_last{lookback}_{datetime.today().replace(microsecond=0)}.csv')

    return trades2

# %%
def get_open_positions(account_id='255715792', write_to_file: bool=False, symbols: list=None):
    open_pos = c.get_account(account_id = account_id, fields=Client.Account.Fields('positions'))
    res = pd.DataFrame(pd.read_json(open_pos).T['positions'].values[0])
    inst = pd.DataFrame(res['instrument'].to_list())
    res = res[['shortQuantity','longQuantity','marketValue','maintenanceRequirement']]
    res['quantity'] = res['longQuantity'] - res['shortQuantity']
    res[['symbol','cusip', 'putCall','underlying']] = inst[['symbol','cusip','putCall','underlyingSymbol']]
    res['strike'] = pd.to_numeric(res['cusip'].str[-7:])/1000

    res = res.drop(['shortQuantity', 'longQuantity', 'cusip'], axis=1).set_index('symbol').sort_index()
    
    if symbols is not None:
        res = res[res['underlying'].isin(symbols)]

    if write_to_file:
        res.to_csv(f'open_positions_{datetime.today().replace(microsecond=0)}.csv')
    
    return res

# %%
def get_spread_trades(filename_html: str, symbols: list or str = None, spread_types: list or str = None, write_to_file: bool=False):

    x = get_trades_raw(filename_html)

    x['Exec Time'] = pd.to_datetime(x['Exec Time'], format='%m/%d/%y %H:%M:%S')

    x2 = x[x['Symbol'].isin(symbols)].drop(['Unnamed: 0','index'], axis=1)

    x2 = x2.fillna(method='ffill').replace('DEBIT',np.nan).replace('CREDIT', np.nan).sort_values(by='Exec Time', ascending=False)

    x3 = x2.copy()

    x3['Net Price'] = pd.to_numeric(x3['Net Price'])

    x3['Main Strike'] = np.where(((x3['Spread'] == 'BUTTERFLY') & (x3['Qty'] > 0)) | ((x3['Spread'] == 'VERTICAL') & (x3['Qty'] < 0)), np.nan, x3['Strike'])

    x3['Main Qty'] = np.where(((x3['Spread'] == 'BUTTERFLY') | (x3['Spread'] == 'VERTICAL')) & (np.isnan(x3['Net Price'])), np.nan, x3['Qty'])

    x3['Cost'] = x3['Net Price']*x3['Main Qty']*100
        
    closes2 = x3[x3['Pos Effect']=='TO CLOSE']

    opens2 = x3[x3['Pos Effect']=='TO OPEN']

    opens3 = opens2.groupby(['Symbol', 'Spread', 'Exp', 'Strike']).agg(exec_time = ('Exec Time', min), qty = ('Qty', sum), cost = ('Cost', sum), 
                                                                    main_strike = ('Main Strike', max), main_qty = ('Main Qty', sum))

    closes3 = closes2.reset_index().set_index(['Symbol', 'Exp', 'Strike']).rename(columns={'Exec Time': 'exec_time', 'Qty': 'qty', 'Main Strike': 'main_strike',
                                                                                        'Main Qty': 'main_qty', 'Cost': 'cost'}) \
                                                                                            .drop(['Side', 'Pos Effect', 'Price', 'Net Price', 'Order Type', 'open_pos', 'index'], axis=1)

    opens3 = opens3.reset_index().set_index(['Symbol', 'Exp', 'Strike'])

    trades = pd.merge(opens3, closes3, how='outer', left_index=True, right_index=True, suffixes=['_open','_close']).reset_index()

    trades['exec_time_close'] = trades['exec_time_close'].where( (trades['exec_time_close'] > trades['exec_time_open']) | (trades['exec_time_close'].isna()), np.nan)

    trades['qty_close'] = trades['qty_close'].where((np.sign(trades['qty_close']) != np.sign(trades['qty_open'])) & 
                                                    (~trades['qty_close'].isna()) & 
                                                    (np.abs(trades['qty_close']) <= np.abs(trades['qty_open'])) &
                                                    (~trades['exec_time_close'].isna()), np.nan)

    trades[['Spread_close', 'Type', 'main_strike_close', 'main_qty_close', 'cost_close']] = \
        trades[['Spread_close', 'Type', 'main_strike_close', 'main_qty_close', 'cost_close']].where((~trades['exec_time_close'].isna()) &
                                                                                                                        (~trades['qty_close'].isna()), np.nan)

    trades = trades.drop_duplicates()

    trades['main_strike_open'] = trades['main_strike_open'].where(~trades['main_strike_open'].isna(), 0)

    trades2 = trades.groupby(['Symbol', 'Exp', 'Strike', 'Spread_open', 'exec_time_open', 'qty_open', 'cost_open', 'main_strike_open', 'main_qty_open']) \
                        .agg(exec_time_close = ('exec_time_close', np.nanmax), qty_close = ('qty_close', np.nansum), cost_close = ('cost_close', np.nansum)) \
                        .reset_index()

    trades2['qty_close'] = trades2['qty_close'].where(trades2['main_qty_open'] > 0, 0)

    trades2 = trades2.sort_values(by=['exec_time_open', 'Strike'], ascending=False)


    trades2['capital'] = np.where((trades2['Spread_open'] == 'BUTTERFLY') & (trades2['cost_open'] < 0), 
                            ((trades2['Strike'] - trades2['Strike'].shift(-1))*100*trades2['main_qty_open'] + trades2['cost_open']),
                            trades2['cost_open'])

    trades3 = trades2[trades2['Spread_open'] != 'SINGLE']

    trades4 = trades3.groupby(['Symbol', 'Exp', 'Spread_open', 'exec_time_open']).agg(main_strike = ('main_strike_open', np.nansum), qty_open = ('main_qty_open', np.nansum), 
                                                                                        cost_open = ('cost_open', np.nansum), cap_at_risk = ('capital', np.nansum),
                                                                                        exec_time_close = ('exec_time_close', np.nanmax), qty_close = ('qty_close', np.nansum), 
                                                                                        cost_close = ('cost_close', np.nansum)).reset_index()

    singles = trades2[trades2['Spread_open'] == 'SINGLE'][['Symbol', 'Exp', 'Spread_open', 'main_strike_open', 'exec_time_open', 'main_qty_open', 'cost_open', 'capital',
                                                        'exec_time_close', 'qty_close', 'cost_close']].rename(columns={'main_strike_open': 'main_strike',
                                                                                                        'main_qty_open': 'qty_open',
                                                                                                        'capital': 'cap_at_risk',
                                                                                                        'qty_close': 'qty_close'})

    trades5 = pd.concat([trades4, singles]).sort_values(by='exec_time_open', ascending=False)

    trades5['exec_time_open'] = trades5['exec_time_open'].dt.date

    trades5['exec_time_close'] = trades5['exec_time_close'].dt.date

    trades5['cost_close'] = trades5['cost_close']*-1

    trades5 = trades5.rename(columns={'Symbol': 'underlying', 'Exp': 'exp', 'Spread_open': 'spread', 'main_strike': 'strike', 'exec_time_open': 'date_open', 'exec_time_close': 'date_close'}).set_index('underlying')

    if spread_types is None or 'BUTTERFLY' in spread_types:
        trades5 = trades5[['spread', 'exp', 'strike', 'date_open', 'qty_open', 'cost_open', 'cap_at_risk', 'date_close', 'qty_close', 'cost_close']]
    else:
        trades5 = trades5[['spread', 'exp', 'strike', 'date_open', 'qty_open', 'cost_open', 'date_close', 'qty_close', 'cost_close']]

    if spread_types is not None:
        trades5 = trades5[trades5['spread'].isin(spread_types)]

    if write_to_file:
        trades5.to_csv(f'spread_trades_{datetime.today().replace(microsecond=0)}.csv')

    return trades5


