
## Initial filtering
Crawl investing.com / whatever for greatest diff between pre-market open and yesterdays close.

## StockEvaluator: CNN || Random Forest, classification
Works as the initial filter, by giving an up/down score based on how much a given security will move
within a daily timeframe.

Input1: daily high, low, volume for the past 3 months.

Output: Score based on probable percentage range

## PriceEvaluator: GRU, regression

Input1: intra-day prices from past two weeks
Input2: Prices from pre-market

Output: Entry and Exit with the greatest diff for the given day. 

## ReEvaluator: Random Forest
Adapt to changing circumstances.

## RiskManager
Act as a check of the different models


## Logical flow
initial filter -> stock evaluator -> price evaluator -> trade state-thread (ReEvaluator).