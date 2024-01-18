from option_pricing_binomial_tree import BinomialOptionTree

excercise_type = 'american'
type = 'call'
position = 'buy'
n = 3
volatility = 0.3
r = 0.1
delta = 0.0
k = 41.0
s = 40.0

plot = True
text_trees = True

# Full tree mode
option_full_tree = BinomialOptionTree('full', option_excercise_type = excercise_type, option_type = type, position = position, n = n, volatility = volatility, r = r, delta = delta, k = k, s = s)
option_full_tree.return_tree(plot = plot, text_trees = text_trees)
print(f'Valuation day option price using full tree: {option_full_tree.option_price[0][0]}')

# 'lite' tree mode
option_lite_tree = BinomialOptionTree('lite', option_excercise_type = excercise_type, option_type = type, position = position, n = n, volatility = volatility, r = r, delta = delta, k = k, s = s)
option_lite_tree.return_tree()
print(f'Valuation day option price using lite tree: {option_lite_tree.option_price}')