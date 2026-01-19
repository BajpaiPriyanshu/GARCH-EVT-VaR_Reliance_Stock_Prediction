import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from arch import arch_model
from scipy import stats
from scipy.stats import genpareto
import warnings
warnings.filterwarnings('ignore')

import sys
import io
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

np.random.seed(42)

print("="*80)
print("GARCH-EVT HYBRID MODEL FOR VALUE AT RISK CALCULATION")
print("="*80)

print("\n[PHASE 1] Data Acquisition")
print("-" * 80)

ticker = "RELIANCE.NS"
target_date = "2026-01-01"
lookback_years = 5

target_dt = pd.to_datetime(target_date)
start_date = (target_dt - pd.DateOffset(years=lookback_years)).strftime('%Y-%m-%d')

print(f"Ticker: {ticker}")
print(f"Data Range: {start_date} to {target_date}")
print(f"Downloading historical data...")

stock = yf.Ticker(ticker)
hist_data = stock.history(start=start_date, end=target_date)

prices = hist_data['Close'].dropna()

print(f"Total observations: {len(prices)}")
print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"First price: Rs{prices.iloc[0]:.2f}")
print(f"Last price: Rs{prices.iloc[-1]:.2f}")

print("\n[PHASE 2] Return Calculation and Visualization")
print("-" * 80)

returns = np.log(prices / prices.shift(1)) * 100
returns = returns.dropna()

print(f"Returns calculated: {len(returns)} observations")
print(f"\nDescriptive Statistics:")
print(f"  Mean: {returns.mean():.4f}%")
print(f"  Std Dev: {returns.std():.4f}%")
print(f"  Skewness: {returns.skew():.4f}")
print(f"  Kurtosis: {returns.kurtosis():.4f} (excess)")
print(f"  Min: {returns.min():.4f}%")
print(f"  Max: {returns.max():.4f}%")

plt.figure(figsize=(14, 6))
plt.subplot(2, 1, 1)
plt.plot(returns.index, returns.values, linewidth=0.8, color='darkblue', alpha=0.7)
plt.title(f'{ticker} - Daily Log Returns (Volatility Clustering Visible)', fontsize=12, fontweight='bold')
plt.ylabel('Return (%)')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='red', linestyle='--', linewidth=0.8)

plt.subplot(2, 1, 2)
plt.hist(returns.values, bins=100, color='steelblue', edgecolor='black', alpha=0.7)
plt.axvline(returns.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean: {returns.mean():.2f}%')
plt.title('Return Distribution', fontsize=12, fontweight='bold')
plt.xlabel('Return (%)')
plt.ylabel('Frequency')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('returns_visualization.png', dpi=300, bbox_inches='tight')
print("\n✓ Returns visualization saved as 'returns_visualization.png'")

print("\n[PHASE 3] GARCH(1,1) Model Estimation")
print("-" * 80)

garch_model = arch_model(returns, vol='Garch', p=1, q=1, mean='constant', dist='normal')
garch_fit = garch_model.fit(disp='off')

print("GARCH(1,1) Model Summary:")
print(garch_fit.summary())

conditional_vol = garch_fit.conditional_volatility

standardized_residuals = returns / conditional_vol
standardized_residuals = standardized_residuals.dropna()

print(f"\nStandardized Residuals Statistics:")
print(f"  Mean: {standardized_residuals.mean():.4f}")
print(f"  Std Dev: {standardized_residuals.std():.4f}")
print(f"  Skewness: {standardized_residuals.skew():.4f}")
print(f"  Kurtosis: {standardized_residuals.kurtosis():.4f} (excess)")

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

axes[0].plot(returns.index, returns.values, linewidth=0.8, alpha=0.6, label='Returns', color='blue')
axes[0].plot(conditional_vol.index, conditional_vol.values, linewidth=1.5, label='Conditional Volatility (σ_t)', color='red')
axes[0].plot(conditional_vol.index, -conditional_vol.values, linewidth=1.5, color='red')
axes[0].set_title('Returns and GARCH(1,1) Conditional Volatility', fontweight='bold')
axes[0].set_ylabel('Return / Volatility (%)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

axes[1].plot(standardized_residuals.index, standardized_residuals.values, linewidth=0.8, color='green', alpha=0.7)
axes[1].axhline(y=0, color='black', linestyle='--', linewidth=0.8)
axes[1].set_title('Standardized Residuals (z_t = r_t / σ_t)', fontweight='bold')
axes[1].set_ylabel('Standardized Residuals')
axes[1].grid(True, alpha=0.3)

axes[2].hist(standardized_residuals.values, bins=100, color='purple', edgecolor='black', alpha=0.7, density=True)
x_range = np.linspace(standardized_residuals.min(), standardized_residuals.max(), 100)
axes[2].plot(x_range, stats.norm.pdf(x_range, 0, 1), 'r--', linewidth=2, label='Standard Normal')
axes[2].set_title('Distribution of Standardized Residuals vs. Standard Normal', fontweight='bold')
axes[2].set_xlabel('Standardized Residuals')
axes[2].set_ylabel('Density')
axes[2].legend()
axes[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('garch_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ GARCH analysis visualization saved as 'garch_analysis.png'")

print("\n[PHASE 4] Extreme Value Theory - Peaks Over Threshold (POT)")
print("-" * 80)

losses = -standardized_residuals.values

threshold_percentile = 90
threshold = np.percentile(losses, threshold_percentile)

excesses = losses[losses > threshold] - threshold

print(f"Threshold Selection:")
print(f"  Percentile: {threshold_percentile}th")
print(f"  Threshold value (u): {threshold:.4f}")
print(f"  Number of exceedances: {len(excesses)}")
print(f"  Percentage of data exceeding threshold: {100*len(excesses)/len(losses):.2f}%")

print("\nFitting Generalized Pareto Distribution...")

gpd_params = genpareto.fit(excesses, floc=0)
xi_gpd = gpd_params[0]
loc_gpd = gpd_params[1]
sigma_gpd = gpd_params[2]

print(f"\nGPD Parameters:")
print(f"  Shape (ξ): {xi_gpd:.4f}")
print(f"  Location: {loc_gpd:.4f}")
print(f"  Scale (σ): {sigma_gpd:.4f}")

if xi_gpd > 0.5:
    tail_type = "Very heavy-tailed (finite variance may not exist)"
elif xi_gpd > 0:
    tail_type = "Heavy-tailed (Fréchet domain)"
elif xi_gpd == 0:
    tail_type = "Exponential tail (Gumbel domain)"
else:
    tail_type = "Light-tailed with finite upper bound (Weibull domain)"

print(f"  Tail interpretation: {tail_type}")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0, 0].plot(standardized_residuals.index, -standardized_residuals.values, 
                linewidth=0.8, alpha=0.6, color='blue', label='Losses (-z_t)')
axes[0, 0].axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'Threshold (u={threshold:.2f})')
axes[0, 0].scatter(standardized_residuals.index[-len(losses):][losses > threshold], 
                   losses[losses > threshold], color='red', s=20, alpha=0.7, label='Exceedances')
axes[0, 0].set_title('Threshold Exceedances Over Time', fontweight='bold')
axes[0, 0].set_ylabel('Loss')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].hist(excesses, bins=50, density=True, alpha=0.7, color='steelblue', edgecolor='black', label='Empirical')
x_fit = np.linspace(0, excesses.max(), 100)
axes[0, 1].plot(x_fit, genpareto.pdf(x_fit, xi_gpd, loc_gpd, sigma_gpd), 
                'r-', linewidth=2, label='GPD Fit')
axes[0, 1].set_title('Excesses Over Threshold with GPD Fit', fontweight='bold')
axes[0, 1].set_xlabel('Excess')
axes[0, 1].set_ylabel('Density')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

theoretical_quantiles = genpareto.ppf(np.linspace(0.01, 0.99, len(excesses)), xi_gpd, loc_gpd, sigma_gpd)
empirical_quantiles = np.sort(excesses)
axes[1, 0].scatter(theoretical_quantiles, empirical_quantiles, alpha=0.6, s=20)
axes[1, 0].plot([theoretical_quantiles.min(), theoretical_quantiles.max()], 
                [theoretical_quantiles.min(), theoretical_quantiles.max()], 
                'r--', linewidth=2, label='Perfect Fit')
axes[1, 0].set_title('QQ-Plot: GPD Fit Quality', fontweight='bold')
axes[1, 0].set_xlabel('Theoretical Quantiles')
axes[1, 0].set_ylabel('Empirical Quantiles')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

thresholds_test = np.percentile(losses, np.linspace(80, 95, 20))
mean_excesses = []
for u_test in thresholds_test:
    exc_test = losses[losses > u_test] - u_test
    if len(exc_test) > 0:
        mean_excesses.append(exc_test.mean())
    else:
        mean_excesses.append(np.nan)

axes[1, 1].plot(thresholds_test, mean_excesses, 'o-', linewidth=2, markersize=6, color='darkgreen')
axes[1, 1].axvline(x=threshold, color='red', linestyle='--', linewidth=2, label=f'Selected u={threshold:.2f}')
axes[1, 1].set_title('Mean Excess Plot (Threshold Validation)', fontweight='bold')
axes[1, 1].set_xlabel('Threshold (u)')
axes[1, 1].set_ylabel('Mean Excess')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('evt_analysis.png', dpi=300, bbox_inches='tight')
print("\n✓ EVT analysis visualization saved as 'evt_analysis.png'")

print("\n[PHASE 5] Value at Risk (VaR) and Expected Shortfall (ES) Calculation")
print("-" * 80)

forecast_horizon = 1
garch_forecast = garch_fit.forecast(horizon=forecast_horizon, reindex=False)
forecasted_volatility = np.sqrt(garch_forecast.variance.values[-1, 0])

print(f"GARCH Volatility Forecast:")
print(f"  1-day ahead σ_{target_date}: {forecasted_volatility:.4f}%")

confidence_level = 0.99
alpha = 1 - confidence_level

n_exceedances = len(excesses)
n_total = len(losses)
prob_exceed_threshold = n_exceedances / n_total

print(f"\nEVT Quantile Calculation:")
print(f"  Confidence level: {confidence_level*100}%")
print(f"  Alpha (tail probability): {alpha}")
print(f"  P(Loss > u): {prob_exceed_threshold:.4f}")

if prob_exceed_threshold > alpha:
    conditional_prob = alpha / prob_exceed_threshold
    
    if xi_gpd != 0:
        gpd_quantile = threshold + (sigma_gpd / xi_gpd) * ((conditional_prob)**(-xi_gpd) - 1)
    else:
        gpd_quantile = threshold + sigma_gpd * np.log(1 / conditional_prob)
    
    print(f"  GPD quantile (standardized): {gpd_quantile:.4f}")
    
    var_99 = forecasted_volatility * gpd_quantile
    
    if xi_gpd < 1 and xi_gpd != 0:
        es_standardized = (gpd_quantile / (1 - xi_gpd)) + ((sigma_gpd - xi_gpd * threshold) / (1 - xi_gpd))
        es_99 = forecasted_volatility * es_standardized
    else:
        es_standardized = gpd_quantile * 1.2
        es_99 = forecasted_volatility * es_standardized
    
    print(f"  Expected Shortfall quantile (standardized): {es_standardized:.4f}")
    
else:
    print(f"\n⚠ WARNING: Threshold too high. P(Loss > u) = {prob_exceed_threshold:.4f} < α = {alpha}")
    print("  Falling back to empirical quantile...")
    gpd_quantile = np.percentile(losses, confidence_level * 100)
    var_99 = forecasted_volatility * gpd_quantile
    es_99 = forecasted_volatility * np.mean(losses[losses > gpd_quantile])

print("\n" + "="*80)
print("FINAL RISK MEASURES")
print("="*80)

print(f"\n📊 Asset: {ticker}")
print(f"📅 Forecast Date: {target_date}")
print(f"🎯 Confidence Level: {confidence_level*100}%")
print(f"⏱  Horizon: 1 day")

print(f"\n{'─'*80}")
print(f"💰 VALUE AT RISK (VaR)")
print(f"{'─'*80}")
print(f"  On {target_date}, the 1-day {confidence_level*100}% VaR for {ticker} is {var_99:.4f}%")
print(f"\n  Interpretation: There is a {confidence_level*100}% probability that the 1-day loss")
print(f"  will not exceed {var_99:.4f}% (or Rs{var_99/100 * prices.iloc[-1]:.2f} per share)")
print(f"  Equivalently: There is a {alpha*100}% probability of losing more than {var_99:.4f}%")

print(f"\n{'─'*80}")
print(f"📉 EXPECTED SHORTFALL (ES / CVaR)")
print(f"{'─'*80}")
print(f"  On {target_date}, the 1-day {confidence_level*100}% ES for {ticker} is {es_99:.4f}%")
print(f"\n  Interpretation: Given that losses exceed the VaR threshold, the expected")
print(f"  loss is {es_99:.4f}% (or Rs{es_99/100 * prices.iloc[-1]:.2f} per share)")
print(f"  ES is always greater than or equal to VaR: ES/VaR ratio = {es_99/var_99:.2f}")

print(f"\n{'─'*80}")
print(f"📈 MODEL PARAMETERS SUMMARY")
print(f"{'─'*80}")
print(f"  GARCH(1,1) Parameters:")
print(f"    ω (omega): {garch_fit.params['omega']:.6f}")
print(f"    α (alpha): {garch_fit.params['alpha[1]']:.6f}")
print(f"    β (beta): {garch_fit.params['beta[1]']:.6f}")
print(f"    Persistence (α+β): {garch_fit.params['alpha[1]'] + garch_fit.params['beta[1]']:.6f}")
print(f"\n  GPD Parameters:")
print(f"    ξ (shape): {xi_gpd:.4f}")
print(f"    σ (scale): {sigma_gpd:.4f}")
print(f"    u (threshold): {threshold:.4f} ({threshold_percentile}th percentile)")
print(f"    Tail type: {tail_type}")

print(f"\n{'─'*80}")
print("GENERATING FINAL VISUALIZATION")
print(f"{'─'*80}")

fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

ax1 = fig.add_subplot(gs[0, :])
ax1.plot(returns.index, returns.values, linewidth=0.8, alpha=0.6, color='blue', label='Daily Returns')
ax1.axhline(y=-var_99, color='red', linestyle='--', linewidth=2, label=f'99% VaR = {var_99:.2f}%')
ax1.axhline(y=-es_99, color='darkred', linestyle=':', linewidth=2, label=f'99% ES = {es_99:.2f}%')

exceedances_idx = returns < -var_99
ax1.scatter(returns.index[exceedances_idx], returns[exceedances_idx], 
           color='red', s=50, alpha=0.8, zorder=5, label=f'VaR Exceedances ({exceedances_idx.sum()})')

ax1.set_title(f'{ticker} - Returns with 99% VaR and ES Thresholds', fontsize=14, fontweight='bold')
ax1.set_ylabel('Return (%)')
ax1.legend(loc='lower left')
ax1.grid(True, alpha=0.3)

ax2 = fig.add_subplot(gs[1, 0])
ax2.plot(conditional_vol.index, conditional_vol.values, linewidth=1.2, color='purple', label='GARCH Volatility')
ax2.axhline(y=forecasted_volatility, color='red', linestyle='--', linewidth=2, 
           label=f'Forecast: {forecasted_volatility:.2f}%')
ax2.set_title('GARCH(1,1) Conditional Volatility', fontsize=12, fontweight='bold')
ax2.set_ylabel('Volatility (%)')
ax2.legend()
ax2.grid(True, alpha=0.3)

ax3 = fig.add_subplot(gs[1, 1])
ax3.plot(standardized_residuals.index, -standardized_residuals.values, 
        linewidth=0.8, alpha=0.6, color='green', label='Losses (-z_t)')
ax3.axhline(y=threshold, color='red', linestyle='--', linewidth=2, label=f'EVT Threshold')
ax3.scatter(standardized_residuals.index[-len(losses):][losses > threshold], 
           losses[losses > threshold], color='red', s=15, alpha=0.7)
ax3.set_title('Standardized Residuals with EVT Threshold', fontsize=12, fontweight='bold')
ax3.set_ylabel('Standardized Loss')
ax3.legend()
ax3.grid(True, alpha=0.3)

ax4 = fig.add_subplot(gs[2, 0])
ax4.hist(excesses, bins=30, density=True, alpha=0.7, color='steelblue', edgecolor='black', label='Empirical')
x_fit = np.linspace(0, excesses.max(), 200)
ax4.plot(x_fit, genpareto.pdf(x_fit, xi_gpd, loc_gpd, sigma_gpd), 
        'r-', linewidth=2.5, label='GPD Fit')
ax4.set_title('Generalized Pareto Distribution Fit', fontsize=12, fontweight='bold')
ax4.set_xlabel('Excess Over Threshold')
ax4.set_ylabel('Density')
ax4.legend()
ax4.grid(True, alpha=0.3)

ax5 = fig.add_subplot(gs[2, 1])
ax5.axis('off')

summary_text = f"""
╔══════════════════════════════════════════╗
║      RISK METRICS SUMMARY                ║
╚══════════════════════════════════════════╝

Asset: {ticker}
Forecast Date: {target_date}
Confidence: {confidence_level*100}%

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
VALUE AT RISK (VaR)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
99% 1-day VaR: {var_99:.4f}%

Maximum expected loss (99% confidence):
  • Percentage: {var_99:.4f}%
  • Rupees/share: Rs{var_99/100 * prices.iloc[-1]:.2f}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
EXPECTED SHORTFALL (ES)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
99% 1-day ES: {es_99:.4f}%

Expected loss beyond VaR:
  • Percentage: {es_99:.4f}%
  • Rupees/share: Rs{es_99/100 * prices.iloc[-1]:.2f}
  • ES/VaR ratio: {es_99/var_99:.2f}x

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
MODEL COMPONENTS
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
GARCH Forecast: σ = {forecasted_volatility:.4f}%
EVT Shape (ξ): {xi_gpd:.4f}
EVT Scale (σ): {sigma_gpd:.4f}
Threshold: {threshold:.4f} ({threshold_percentile}th %ile)
Historical Exceedances: {exceedances_idx.sum()} / {len(returns)}
Expected Exceedances: {int(len(returns) * alpha)}
"""
ax5.text(0.05, 0.95, summary_text, transform=ax5.transAxes,
fontsize=9, verticalalignment='top', fontfamily='monospace',
bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

plt.suptitle(f'GARCH-EVT Hybrid Model: Complete Risk Analysis for {ticker}',
fontsize=16, fontweight='bold', y=0.995)
plt.savefig('final_var_analysis.png', dpi=300, bbox_inches='tight')
print("✓ Final comprehensive visualization saved as 'final_var_analysis.png'")

print(f"\n{'='*80}")
print("BACKTESTING RESULTS")
print(f"{'='*80}")

actual_exceedances = exceedances_idx.sum()
expected_exceedances = len(returns) * alpha
exceedance_rate = actual_exceedances / len(returns)

print(f"\nVaR Backtesting (Historical Period):")
print(f"  Total observations: {len(returns)}")
print(f"  Actual exceedances: {actual_exceedances}")
print(f"  Expected exceedances: {expected_exceedances:.1f}")
print(f"  Actual exceedance rate: {exceedance_rate*100:.2f}%")
print(f"  Expected exceedance rate: {alpha*100:.2f}%")

if abs(exceedance_rate - alpha) < 0.005:
    zone = "GREEN ZONE ✓"
    assessment = "Model performs well"
elif abs(exceedance_rate - alpha) < 0.01:
    zone = "YELLOW ZONE ⚠"
    assessment = "Model acceptable, monitor closely"
else:
    zone = "RED ZONE ✗"
    assessment = "Model may need recalibration"

print(f"\n  Backtesting Assessment: {zone}")
print(f"  {assessment}")

print(f"\n{'='*80}")
print("ANALYSIS COMPLETE")
print(f"{'='*80}")
print("\n📁 Output Files Generated:")
print("  1. returns_visualization.png - Returns series and distribution")
print("  2. garch_analysis.png - GARCH model results and diagnostics")
print("  3. evt_analysis.png - EVT fitting and validation")
print("  4. final_var_analysis.png - Comprehensive risk analysis dashboard")
print("\n✅ All calculations completed successfully!")
print("="*80)