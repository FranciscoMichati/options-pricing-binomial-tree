from option_pricing_binomial_tree import BinomialOptionTree

# Define the main function
def main() -> None:
    # Configuration Parameters
    exercise_type: str = 'american'
    option_type: str = 'call'
    position: str = 'buy'
    n: int = 3
    volatility: float = 0.3
    risk_free_rate: float = 0.1
    delta: float = 0.0
    strike_price: float = 41.0
    initial_stock_price: float = 40.0

    # Output Options
    plot_tree: bool = True
    display_text_trees: bool = True

    # Full Tree Mode
    option_full_tree = BinomialOptionTree(
        'full',
        option_excercise_type=exercise_type,
        option_type=option_type,
        position=position,
        n=n,
        volatility=volatility,
        r=risk_free_rate,
        delta=delta,
        k=strike_price,
        s=initial_stock_price
    )
    option_full_tree.return_tree(plot=plot_tree, text_trees=display_text_trees)
    print(f'Valuation day option price using full tree: {option_full_tree.option_price[0][0]}')

    # Lite Tree Mode
    option_lite_tree = BinomialOptionTree(
        'lite',
        option_excercise_type=exercise_type,
        option_type=option_type,
        position=position,
        n=n,
        volatility=volatility,
        r=risk_free_rate,
        delta=delta,
        k=strike_price,
        s=initial_stock_price
    )
    option_lite_tree.return_tree()
    print(f'Valuation day option price using lite tree: {option_lite_tree.option_price}')

# Execute main function
if __name__ == '__main__':
    main()
