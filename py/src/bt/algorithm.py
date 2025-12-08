from zipline.api import *
import pandas as pd
from src.utils import get_ols_fit_model


class PairsTradingAlgorithm(TradingAlgorithm):
    def initialize(self, strategy):
        self.strategy = strategy
        self.assets = [symbol_lookup(symbol) for symbol in strategy.symbols]
        self.training_start = pd.Timestamp(strategy.training_start)
        self.training_end = pd.Timestamp(strategy.training_end)
        self.retrain_interval = pd.offsets.MonthEnd(strategy.retrain_interval_months)
        self.last_retrain = self.training_end
        self.alpha = None
        self.beta = None
        self.mean_spread = None
        self.std_spread = None
        # Initial retrain
        self.retrain()

    def retrain(self):
        # Get historical data from training_start to last_retrain
        days = (self.last_retrain - self.training_start).days
        hist = self.history(self.assets, 'close', bar_count=days, frequency='1d')
        s1 = hist[self.assets[0]]
        s2 = hist[self.assets[1]]
        model = get_ols_fit_model(s1, s2)
        self.alpha, self.beta = model.params
        spread_series = s1 - (self.alpha + self.beta * s2)
        self.mean_spread = spread_series.mean()
        self.std_spread = spread_series.std()
        # Update for next retrain
        self.last_retrain += self.retrain_interval

    def handle_data(self, data):
        if self.datetime >= self.last_retrain:
            self.retrain()
        # Calculate z_score
        close1 = data.current(self.assets[0], 'close')
        close2 = data.current(self.assets[1], 'close')
        scaled_s2 = self.alpha + self.beta * close2
        spread = close1 - scaled_s2
        z_score = (spread - self.mean_spread) / self.std_spread
        # Check for entry
        if abs(z_score) > self.strategy.entry_z and not self.portfolio.positions:
            if z_score < -self.strategy.entry_z:
                self.order(self.assets[0], self.strategy.position_size * self.portfolio.portfolio_value)
                self.order(self.assets[1], -self.strategy.position_size * self.portfolio.portfolio_value)
            elif z_score > self.strategy.entry_z:
                self.order(self.assets[0], -self.strategy.position_size * self.portfolio.portfolio_value)
                self.order(self.assets[1], self.strategy.position_size * self.portfolio.portfolio_value)
        # Check for exit
        if self.portfolio.positions:
            # Simple exit when z crosses 0
            if z_score * self.strategy.entry_z < 0:
                for asset in self.assets:
                    if asset in self.portfolio.positions:
                        self.order(asset, 0)
        self.record(z_score=z_score)</content>
<parameter name="filePath">src/bt/algorithm.py