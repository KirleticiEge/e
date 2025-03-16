import pandas as pd
import yfinance as yf
import time
from collections import Counter
import matplotlib.pyplot as plt

# PairsTradingML sınıfını içeren TEST3.py dosyasından içe aktarın
from TEST3 import PairsTradingML

# -----------------------
# Yfinance ile toplu veri indirme ve temizleme
# -----------------------
start_date = "2015-01-01"
end_date   = "2025-03-15"
data_frequency = "1d"

ticker1="AKBNK.IS"
ticker2="GARAN.IS"

tickers = ['AEFES.IS', 'AGHOL.IS', 'AGROT.IS', 'AKBNK.IS', 'AKFYE.IS', 'AKSA.IS', 'AKSEN.IS', 
           'ALARK.IS', 'ALFAS.IS', 'ALTNY.IS', 'ANHYT.IS', 'ANSGR.IS', 'ARCLK.IS', 'ARDYZ.IS', 
           'ASELS.IS', 'ASTOR.IS', 'BERA.IS', 'BIMAS.IS', 'BRSAN.IS', 'BRYAT.IS', 'BSOKE.IS', 
           'BTCIM.IS', 'CANTE.IS', 'CCOLA.IS', 'CIMSA.IS', 'CLEBI.IS', 'CVKMD.IS', 'CWENE.IS', 
           'DOAS.IS', 'DOHOL.IS', 'ECILC.IS', 'EGEEN.IS', 'EKGYO.IS', 'ENERY.IS', 'ENJSA.IS', 
           'ENKAI.IS', 'EREGL.IS', 'EUPWR.IS', 'FENER.IS', 'FROTO.IS', 'GARAN.IS', 'GESAN.IS', 
           'GOLTS.IS', 'GUBRF.IS', 'HALKB.IS', 'HEKTS.IS', 'IEYHO.IS', 'ISCTR.IS', 'ISMEN.IS', 
           'KARSN.IS', 'KCAER.IS', 'KCHOL.IS', 'KLSER.IS', 'KONTR.IS', 'KONYA.IS', 'KOZAA.IS', 
           'KOZAL.IS', 'KRDMD.IS', 'LIDER.IS', 'MAGEN.IS', 'MAVI.IS', 'MGROS.IS', 'MIATK.IS', 
           'MPARK.IS', 'NTHOL.IS', 'ODAS.IS', 'OTKAR.IS', 'OYAKC.IS', 'PASEU.IS', 'PETKM.IS', 
           'PGSUS.IS', 'REEDR.IS', 'SAHOL.IS', 'SASA.IS', 'SDTTR.IS', 'SELEC.IS', 'SISE.IS', 
           'SKBNK.IS', 'SMRTG.IS', 'SOKM.IS', 'TABGD.IS', 'TAVHL.IS', 'TCELL.IS', 'THYAO.IS', 
           'TKFEN.IS', 'TMSN.IS', 'TOASO.IS', 'TSKB.IS', 'TSPOR.IS', 'TTKOM.IS', 'TTRAK.IS', 
           'TUKAS.IS', 'TUPRS.IS', 'TURSG.IS', 'ULKER.IS', 'VAKBN.IS', 'VESTL.IS', 'YEOTK.IS', 
           'YKBNK.IS', 'ZOREN']

print("Starting bulk download for tickers...")
data = yf.download(tickers, start=start_date, end=end_date, interval=data_frequency, group_by='ticker')

# Hangi tickerların başarılı indirildiğini belirleyelim
downloaded_tickers = []
for ticker in tickers:
    if ticker in data.columns.get_level_values(0):
        ticker_df = data[ticker]
        if not ticker_df['Close'].dropna().empty:
            downloaded_tickers.append(ticker)

failed_tickers = [ticker for ticker in tickers if ticker not in downloaded_tickers]
print(f"Bulk download complete: {len(downloaded_tickers)} tickers downloaded, {len(failed_tickers)} missing.")

# Her ticker için fiyat serilerini oluşturuyoruz
price_series = {}
for ticker in downloaded_tickers:
    df_ticker = data[ticker]
    if 'Adj Close' in df_ticker.columns:
        series = df_ticker['Adj Close']
    elif 'Close' in df_ticker.columns:
        series = df_ticker['Close']
    else:
        continue
    series = series.squeeze()
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=df_ticker.index)
    price_series[ticker] = series

# Eksik tickerları tek tek indiriyoruz
for ticker in failed_tickers:
    df_ticker = yf.download(ticker, start=start_date, end=end_date, interval=data_frequency)
    if df_ticker.empty:
        continue
    if 'Adj Close' in df_ticker.columns:
        series = df_ticker['Adj Close']
    elif 'Close' in df_ticker.columns:
        series = df_ticker['Close']
    else:
        continue
    series = series.squeeze()
    if not isinstance(series, pd.Series):
        series = pd.Series(series, index=df_ticker.index)
    price_series[ticker] = series
    time.sleep(1)

# Tüm fiyat serilerini tek DataFrame'de birleştiriyoruz
price_df = pd.concat(price_series, axis=1)

# Temizlik: indexlerin uyumlu olduğunu varsayıyoruz; eksik verileri kontrol edip, temizliyoruz
ref_index = price_df.iloc[:, 0].index
tickers_missing = {ticker: price_df[ticker].isnull().sum() for ticker in price_df.columns if price_df[ticker].isnull().sum() > 1}
print(tickers_missing)

tickers_one_day_missing = [ticker for ticker in price_df.columns if price_df[ticker].isnull().sum() == 1]
print(len(tickers_one_day_missing), "tickers have exactly one missing day.")

from collections import Counter
missing_dates = [price_df[ticker][price_df[ticker].isnull()].index[0] for ticker in tickers_one_day_missing]
missing_date_counts = Counter(missing_dates)
print("Missing day counts:")
for day, count in missing_date_counts.items():
    print(f"{day}: {count} tickers")

tickers_to_keep = [ticker for ticker in price_df.columns if price_df[ticker].isnull().sum() <= 1]
filtered_df = price_df[tickers_to_keep]
print(f"Stocks kept: {len(tickers_to_keep)} out of {price_df.shape[1]}")

clean_price_df = filtered_df.dropna(axis=0, how='any')
removed_days = filtered_df.shape[0] - clean_price_df.shape[0]
print(f"Removed {removed_days} days with missing data.")
print("Final cleaned data shape:", clean_price_df.shape)

# Opsiyonel: Günlük getirileri çizdirelim
returns_df = clean_price_df.pct_change().dropna()
returns_df.drop(columns="TKFEN.IS", inplace=True)
plt.figure(figsize=(12, 6))
for asset in returns_df.columns:
    plt.plot(returns_df.index, returns_df[asset], label=asset)
plt.title('Daily Returns of Selected Assets')
plt.xlabel('Date')
plt.ylabel('Returns')
plt.legend()
plt.grid(True)
plt.show()

# -----------------------
# PairsTradingML için veri aktarımı
# -----------------------
# TEST3.py'deki PairsTradingML sınıfı, load_price_data() metodu ile CSV'den okuyor.
# Biz burada bulk download ile elde edilen clean_price_df verisini kullanıyoruz.
# Dolayısıyla ptml.price_data sözlüğünü, clean_price_df'deki her sütunu tek sütunlu DataFrame olarak dolduruyoruz.

ptml = PairsTradingML(data_folder="dummy", output_folder="ML_Models")
ptml.price_data = {}  # Kendi indirilen verimizi kullanacağız
for ticker in clean_price_df.columns:
    df = pd.DataFrame(clean_price_df[ticker])
    df.columns = [ticker]
    df.index = pd.to_datetime(df.index)
    ptml.price_data[ticker] = df

# Stationary çiftleri CSV'den yüklemeye gerek kalmadığı için load_stationary_pairs() çağrısını kaldırıyoruz.
# Bunun yerine, örnek olarak tek bir çift (örneğin "ISCTR" ve "YKBNK") için stationary çiftleri manuel olarak belirliyoruz.
ptml.stationary_pairs = pd.DataFrame({
    "Ticker1": [ticker1],
    "Ticker2": [ticker2],
    "Hedge_Ratio": [1.0]  # Uygun hedge oranını burada belirleyin
})

# Diğer işlemler: verinin hazırlanması, gelişmiş workflow, model eğitimi, backtest ve görselleştirme
ptml.prepare_pair_data()
ptml.run_enhanced_workflow(ticker1=ticker1, ticker2=ticker2)
ptml.train_all_models()
df_backtest = ptml.backtest_strategy(ticker1, ticker2)
ptml.plot_strategy_performance(df_backtest, ticker1, ticker2)
