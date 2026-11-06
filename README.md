# KRX Stock Trading Algorithm

Entry for the **KRX (Korea Exchange) stock trading algorithm competition** —
reached the finals (본선 진출).

The competition ran in two stages: a qualifying round judged on submitted code,
and a final round judged on a written algorithm specification defending the
strategy.

---

## Contents

| Stage | Item | File |
| --- | --- | --- |
| Qualifier | Trading algorithm notebook | [`krx-trading-algorithm.ipynb`](krx-trading-algorithm.ipynb) |
| Final | Algorithm specification (Korean, PDF) | [`investment-algorithm-specification.pdf`](investment-algorithm-specification.pdf) |

## The approach

The notebook builds a quantitative strategy over KRX-listed equities: price and
volume data is assembled, technical indicators and engineered features are
derived from it, and signals drive a backtested rule set. The specification PDF
documents the reasoning, the indicator choices and the risk handling behind the
submitted algorithm.

## Running it

```bash
git clone https://github.com/paulms77/KRX.git
cd KRX

pip install pandas numpy FinanceDataReader pykrx matplotlib scikit-learn
jupyter notebook krx-trading-algorithm.ipynb
```

Market data is pulled through `FinanceDataReader` and `pykrx` at runtime rather
than being stored in the repository.

## Related

| Repository | Relation |
| --- | --- |
| [BiLSTM-StockPrediction-Algorithm](https://github.com/paulms77/BiLSTM-StockPrediction-Algorithm) | The research side project this grew into — bidirectional LSTM price prediction with a paper |
| [Dacon](https://github.com/paulms77/Dacon) | The qualifying notebook also sits with the other DACON entries |
| [bitcoin](https://github.com/paulms77/bitcoin) | RSI-driven automated trading on Upbit |

## License

Competition code — free to reference.
