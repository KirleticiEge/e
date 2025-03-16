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


def calculate_spread_and_zscore(price_df, ticker1, ticker2, window=20, hedge_method='ols'):
    """
    Verilen fiyat verisi DataFrame'inden spread ve z-score hesaplar.
    
    Parameters:
    -----------
    price_df : pandas DataFrame
        Hisse senedi fiyat verileri
    ticker1, ticker2 : str
        Çiftin hisse senedi sembolleri
    window : int
        Z-score hesaplanırken kullanılacak pencere büyüklüğü
    hedge_method : str
        'ols' - Ordinary Least Squares ile hedge oranı hesapla
        'ratio' - Basit fiyat oranı kullan
    
    Returns:
    --------
    pandas DataFrame
        Spread ve z-score değerleri
    """
    # Fiyat serilerini al
    y = price_df[ticker1]
    x = price_df[ticker2]
    
    # Hedge oranı hesaplama
    if hedge_method == 'ols':
        # Eğitim verisinde OLS kullanarak hedge oranı hesapla
        import statsmodels.api as sm
        model = sm.OLS(y, sm.add_constant(x)).fit()
        hedge_ratio = model.params[1]
        alpha = model.params[0]
        
        # Spread = y - (intercept + hedge_ratio * x)
        spread = y - (alpha + hedge_ratio * x)
        
        print(f"OLS Hedge oranı: {hedge_ratio:.4f}, Alpha: {alpha:.4f}")
    else:
        # Basit fiyat oranı
        hedge_ratio = y.iloc[0] / x.iloc[0]
        spread = y - hedge_ratio * x
        
        print(f"Basit oran hesaplama: {hedge_ratio:.4f}")
    
    # Hareketli ortalama ve standart sapma ile Z-score hesaplama
    rolling_mean = spread.rolling(window=window).mean()
    rolling_std = spread.rolling(window=window).std()
    zscore = (spread - rolling_mean) / rolling_std
    
    # Spread ve Z-score için volatilite hesapla
    spread_volatility = spread.rolling(window=window).std()
    zscore_volatility = zscore.rolling(window=window).std()

    # DataFrame oluştur
    df = pd.DataFrame({
        'spread': spread,
        f'zscore_{window}d': zscore,
        'spread_volatility': spread_volatility,
        'zscore_volatility': zscore_volatility,
        'hedge_ratio': hedge_ratio
    })
    
    return df.dropna()  # Boş satırları düşür

def backtest_classical_strategy(df, 
                               zscore_col='zscore_20d', 
                               entry_threshold=2.0, 
                               exit_threshold=0.0,
                               initial_capital=10000, 
                               trade_fraction=0.5,
                               stop_loss_pct=0.1,
                               max_position_hold_days=30):
    """
    Klasik z-score yaklaşımını uygulayan düzeltilmiş backtest fonksiyonu.
    """
    df = df.copy()  # Orijinalini bozmamak için kopya al
    
    # Yeni kolonları başlat
    df['position_classic'] = 0
    df['equity_classic'] = initial_capital
    df['trade_pnl'] = 0
    df['trade_days'] = 0
    
    position = 0
    entry_price = None
    entry_date = None
    equity = initial_capital
    allocated_capital = 0
    
    # Gün gün ilerleyerek z-score'a bakıyoruz
    for i in range(1, len(df)):
        current_date = df.index[i]
        zscore = df.iloc[i][zscore_col]
        spread = df.iloc[i]['spread']
        
        # Pozisyon yokken eğer zscore < -2 => long, zscore > +2 => short açıyoruz
        if position == 0:
            if zscore < -entry_threshold:
                position = 1  # Long pozisyon (spread'in artmasını bekliyoruz)
                entry_price = spread
                entry_date = current_date
                allocated_capital = equity * trade_fraction
            elif zscore > entry_threshold:
                position = -1  # Short pozisyon (spread'in düşmesini bekliyoruz)
                entry_price = spread
                entry_date = current_date
                allocated_capital = equity * trade_fraction
        else:
            # Pozisyonda iken çıkış koşullarını kontrol et
            days_in_position = (current_date - entry_date).days if entry_date is not None else 0
            
            # Pozisyonun kar/zarar durumu
            if position == 1:  # Long pozisyon
                pnl_pct = (spread - entry_price) / abs(entry_price)
            else:  # Short pozisyon
                pnl_pct = (entry_price - spread) / abs(entry_price)
            
            # Stop loss tetiklendi mi?
            stop_loss_triggered = pnl_pct < -stop_loss_pct
            
            # Pozisyonu kapat
            if ((position == 1 and zscore >= exit_threshold) or 
                (position == -1 and zscore <= -exit_threshold) or
                stop_loss_triggered or 
                days_in_position >= max_position_hold_days):
                
                # PnL hesapla
                pnl = allocated_capital * pnl_pct
                
                # PnL'yi kısıtla (max kayıp = sermaye)
                if pnl < -allocated_capital:
                    pnl = -allocated_capital
                
                # Sermayeyi güncelle
                equity += pnl
                
                # Trade bilgisini kaydet
                df.at[df.index[i], 'trade_pnl'] = pnl
                df.at[df.index[i], 'trade_days'] = days_in_position
                
                # Pozisyonu sıfırla
                position = 0
                entry_price = None
                entry_date = None
                allocated_capital = 0
        
        # Her gün için sermaye ve pozisyon bilgisini kaydet
        df.at[df.index[i], 'equity_classic'] = max(0, equity)  # Sermaye negatif olamaz
        df.at[df.index[i], 'position_classic'] = position

    # Son pozisyonun durumunu kontrol et ve kapat (eğer açık pozisyon kaldıysa)
    if position != 0 and entry_price is not None:
        last_spread = df.iloc[-1]['spread']
        if position == 1:
            pnl_pct = (last_spread - entry_price) / abs(entry_price)
        else:
            pnl_pct = (entry_price - last_spread) / abs(entry_price)
        
        pnl = allocated_capital * pnl_pct
        
        # PnL'yi kısıtla (max kayıp = sermaye)
        if pnl < -allocated_capital:
            pnl = -allocated_capital
            
        equity += pnl
        df.at[df.index[-1], 'equity_classic'] = max(0, equity)

    # Performans metrikleri ekle
    df['returns'] = df['equity_classic'].pct_change()
    df['cumulative_returns'] = (1 + df['returns']).cumprod() - 1
    df['drawdown'] = 1 - df['equity_classic'] / df['equity_classic'].cummax()
    
    return df

def calculate_performance_metrics(df):
    """
    Backtest sonuçlarından performans metrikleri hesaplar.
    """
    # Temel metrikleri hesapla
    initial_equity = df['equity_classic'].iloc[0]
    final_equity = df['equity_classic'].iloc[-1]
    
    # Toplam getiri
    total_return = (final_equity / initial_equity - 1) * 100
    
    # Yıllık getiri
    days = (df.index[-1] - df.index[0]).days
    years = days / 365.25
    annual_return = (((final_equity / initial_equity) ** (1 / years)) - 1) * 100 if years > 0 else 0
    
    # Maksimum drawdown
    if 'drawdown' not in df.columns:
        df['drawdown'] = 1 - df['equity_classic'] / df['equity_classic'].cummax()
    max_drawdown = df['drawdown'].max() * 100
    
    # Sharpe oranı (Risk-free rate = 0 varsayımı)
    if 'returns' not in df.columns:
        df['returns'] = df['equity_classic'].pct_change()
    
    daily_returns = df['returns'].dropna()
    if len(daily_returns) > 0:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * (252 ** 0.5)  # 252 trading days
    else:
        sharpe_ratio = 0
    
    # İşlem sayısı (trade_pnl != 0 olan günler)
    if 'trade_pnl' in df.columns:
        trade_count = (df['trade_pnl'] != 0).sum()
        winning_trades = (df['trade_pnl'] > 0).sum()
        losing_trades = (df['trade_pnl'] < 0).sum()
        win_rate = (winning_trades / trade_count * 100) if trade_count > 0 else 0
    else:
        trade_count = winning_trades = losing_trades = win_rate = 0
    
    # Sonuçları sözlükte topla
    metrics = {
        'initial_equity': initial_equity,
        'final_equity': final_equity,
        'total_return_pct': total_return,
        'annual_return_pct': annual_return,
        'max_drawdown_pct': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'trade_count': trade_count,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades,
        'win_rate_pct': win_rate
    }
    
    return metrics

# Ana koddaki düzeltmeler:
# 1) Önce spread ve z-score hesaplayalım
df_clean = calculate_spread_and_zscore(clean_price_df, ticker1, ticker2)

# 2) Klasik backtest'i çalıştıralım
df_with_classic = backtest_classical_strategy(df_clean,
                                             zscore_col='zscore_20d',
                                             entry_threshold=2.0,
                                             exit_threshold=0.0,
                                             initial_capital=10000,
                                             trade_fraction=0.5)

# 3) Sonuçları görselleştirelim
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

plt.figure(figsize=(12, 6))
# Önemli düzeltme: clean_price_df değil, df_with_classic dataframe'ini kullanmalıyız
plt.plot(df_with_classic.index, df_with_classic['equity_classic'], label='Klasik Z-Score Stratejisi')

plt.title('Klasik Z-Score Stratejisi Backtest Sermaye Eğrisi')
plt.xlabel('Tarih')
plt.ylabel('Sermaye (Equity)')
plt.legend()
plt.grid(True)

# Y eksenindeki "e6" formatını kaldır ve tam sayı olarak göster
plt.gca().yaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{int(x)}'))

plt.show()

# Opsiyonel: Performans metrikleri hesaplama
final_equity = df_with_classic['equity_classic'].iloc[-1]
total_return = (final_equity / 10000 - 1) * 100
print(f"Başlangıç sermayesi: 10,000")
print(f"Son sermaye: {final_equity:.2f}")
print(f"Toplam getiri: %{total_return:.2f}")
