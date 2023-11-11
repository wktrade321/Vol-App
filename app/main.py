from flask import Flask
from flask import render_template
#from app import app
from calculations import relative_value as rv
#from calculations import earnings as er
#from calculations import etf_dispersion as etf
from datetime import datetime
import yfinance as yf
from scipy import stats
import pandas as pd
import numpy as np




app = Flask(__name__)



"""
TO-DO: Insert scheduler to run plot_current_ratios 
"""

@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')


@app.route('/dashboard')
def dashboard():

    ratios, term_struct, percentiles, futures_curve = rv.get_current_vix_metrics(return_vals=True, display_vals=False)

    ratios_pctls = pd.DataFrame(ratios.rename('ratio')).join(pd.DataFrame(percentiles.rename('percentile')), how='left', lsuffix='x', rsuffix='y')

    avg_contango = ratios.mean()
    avg_percentile = percentiles.mean()

    ratios = [{'name': index, 'value': row['ratio'], 'percentile': row['percentile']} for index,row in ratios_pctls.iterrows()]

    vix = yf.download('^VIX', progress=False)['Adj Close'].dropna()
    vix_price = vix[-1]
    vix_change = vix.pct_change(1)[-1]*100
    vix_percentile = stats.percentileofscore(vix, vix_price)

    rolling_returns_data = []
    spy = yf.download('SPY', period='6d', interval='1d', progress=False)['Adj Close'].dropna()
    rolling_returns_data.append({'name': 'Rolling 5d Return', 'value': spy.pct_change(5)[-1]*100})
    rolling_returns_data.append({'name': 'Rolling 3d Return', 'value': spy.pct_change(3)[-1]*100})
    rolling_returns_data.append({'name': 'Rolling 1d Return', 'value': spy.pct_change(1)[-1]*100})

    iv_rank_data = []
    uvxy = rv.ndl.get('OPT/UVXY')['iv90']
    vxx = rv.ndl.get('OPT/VXX')['iv90']
    spy = rv.ndl.get('OPT/SPY')['iv90']


    iv_rank_data.append({'name': 'UVXY IV90 Rank', 'value': stats.percentileofscore(uvxy.values[:-1], uvxy.values[-1])})
    iv_rank_data.append({'name': 'VXX IV90 Rank', 'value': stats.percentileofscore(vxx.values[:-1], vxx.values[-1])})
    iv_rank_data.append({'name': 'SPY IV90 Rank', 'value': stats.percentileofscore(spy.values[:-1], spy.values[-1])})



    return render_template('dashboard.html', 
                           vix_price=vix_price, 
                           vix_change=vix_change, 
                           vix_percentile=vix_percentile,
                           ratios=ratios, 
                           avg_contango=avg_contango,
                           avg_percentile=avg_percentile,
                           term_struct=term_struct,
                           rolling_returns_data=rolling_returns_data,
                           iv_rank_data=iv_rank_data)
 




@app.route('/futures')
def futures():

    ratios, term_struct, percentiles, futures_curve = rv.get_current_vix_metrics(return_vals=True, display_vals=False)
    rv.plot_vix_futures_curve(futures_curve=futures_curve, title='static/vix_futures_curve')

    return render_template('index_futures_curve.html')


@app.route('/IV30')
def iv_30():
    return render_template('index_iv30.html')



@app.route('/IV60')
def iv_60():
    return render_template('index_iv60.html')



@app.route('/IV90')
def iv_90():
    return render_template('index_iv90.html')


##Run Main
if __name__ == '__main__':
    app.run(debug=True)