import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from binarytree import build

class BinomialOptionTree():

    """

    Inputs
    ----------------
    mode : [str]
        [Allowed values are lite for returning today's option price or full for returning the entire trees]
    option_excercise_type : [str]
        [Allowed values are european and american for each type of option, respectively]
    option_type : [str]
        [Allowed types are call or put]
    position : [str]
        [Allowed valueas are buy or sell]
    n : [int]
        [time periods]
    volatility : [float]
        [Volatility]
    r : [float]
        [Risk-free interest rate]
    delta : [float]
        [Dividends]
    k : [float]
        [Strike price of the option]
    s : [float]
        [Underliyng asset actual price]
    """
    def __init__(self, mode: str, option_excercise_type: str, option_type: str, position: str, 
                 n: int, volatility: float, r: float, delta: float, k: float, s: float):
        
        self.mode = mode
        
        self.option_excercise_type = option_excercise_type
        self.option_type = option_type
        self.positon = position
        
        self.n = n
        self.volatility = volatility
        self.r = r
        self.delta = delta
        self.k = k
        self.s = s

        self.h = 1/self.n
    

    @staticmethod
    def check_entries(mode, option_excercise_type, option_type, position,n, volatility, r, delta, k, s):
        """
        Function to check variable entries and types.
        """

        if (mode != 'lite') and (mode != 'full') :
            raise Exception('mode must be lite or full.')
        if (option_excercise_type != 'european') and (option_excercise_type != 'american') :
            raise Exception('option_excercise_type must be european or american.')
        if (option_type != 'call') and (option_type!= 'put'):
            raise Exception('option_type must be call or put.')
        if (position != 'buy') and (position!= 'sell'):
            raise Exception('position must be buy or sell.')
        if not isinstance(n, int):
            raise Exception('n must be an integer.')
        if not (isinstance(volatility, float)):
            raise Exception('volatility must be a float.')
        if not (isinstance(r, float)):
            raise Exception('risk-free interest rate r must be a float.')
        if not (isinstance(delta, float)):
            raise Exception('delta must be a float.')
        if not (isinstance(k, (float,int))):
            raise Exception('strike price k must be a float or integer number.')
        if not (isinstance(s, float) or isinstance(s, int) ):
            raise Exception('Underlying price k must be a float or integer number.')
        

    def risk_neutral_prob(self):
        u = np.exp((self.r-self.delta)*self.h + self.volatility*np.sqrt(self.h))
        d = np.exp((self.r-self.delta)*self.h - self.volatility*np.sqrt(self.h))
        return (np.exp((self.r-self.delta)*self.h)-d )/(u-d)
    
    def compute_option_price(self, p: float, price_up: float, price_down: float):
        """
        Method to calculate the option price using the binomial model.

        Inputs
        ----------------
        p : [float]
            [Risk-neutral pseudoprobability that stocks goes up]
        price_up : [float] 
            [cost of the option at time t+h if the price of the underlying asset goes up]
        price_down : [float] 
            [cost of the option at time t+h if the price of the underlying asset goes down]

        Returns
        ----------------
        [float] :
            Price of the option
        """
        return np.exp(-self.r*self.h)*(p*price_up+(1-p)*price_down)
    
    def option_payoff(self, s):
        """
        Computes the option payoff at maturity date.

        Inputs
        ----------------
        s : [float]
            [Asset price]

        Returns
        ----------------
        [float] :
            [Option payoff]
        """
        if self.option_type == 'call':
            if self.positon == 'buy':
                return max(0, s-self.k)
            else:
                return -max(0, s-self.k)
            
        elif self.option_type == 'put': 
            if self.positon == 'buy':
                return max(0, self.k-s)
            else:
                return -max(0, self.k-s)
            
        else:
            raise Exception("Error: option type must be 'call' or 'put'.")

    def price_change(self, up: bool):
        if up:
            return np.exp((self.r-self.delta)*self.h + self.volatility*np.sqrt(self.h))
        else:
            return np.exp((self.r-self.delta)*self.h - self.volatility*np.sqrt(self.h))

    def create_asset_full_tree(self):
        """
        Create the full asset prices tree.

        Returns
        ----------------
        [list[float]] :
            [list of lists containing the information of the tree]
        """

        data_array = []
        for i in range(0,self.n+1):
            data_array.append(np.zeros(2**i))

        data_array[0]=[self.s,] #Initialize the first Node with the actual price

        for i in range(1, len(data_array)):
            for j in range(0, len(data_array[i])):
                if (j%2 == 0):
                    data_array[i][j]=data_array[i-1][(j//2)]*self.price_change(True)
                else:
                    data_array[i][j]=data_array[i-1][(j//2)]*self.price_change(False)

        return data_array
    
    def create_option_full_tree(self, asset_prices: list):
        """
        Create the full options prices tree.


        TODO : The function doesn't take into consideration that some nodes have the same value, and for that, repeat a lot of calculations. This has to be improved in the future.

        Inputs
        ----------------
        asset_prices : [list[float]]
            [Asset prices list]
        

        Returns
        ----------------
        [list[float]] :
            [list of lists containing the information of the tree]
        """
        data_array= []

        for i in range(0, self.n+1):
            data_array.append(np.zeros(2**i))
        
        
        asset_prices_at_maturity = asset_prices[-1]
        for i in range(len(data_array[-1])):        # Initialize the option price at maturity date
            data_array[-1][i] = self.option_payoff(asset_prices_at_maturity[i])

        p = self.risk_neutral_prob() 
 
        for i in reversed(range(0, len(data_array)-1)):
            for j in range(0, len(data_array[i])):
                for k in range(0, j+1):
                    price_up = data_array[i+1][2*k]
                    price_down = data_array[i+1][2*k+1]
                if self.option_excercise_type == 'european':
                    data_array[i][j] = self.compute_option_price(p, price_up, price_down)
                elif self.option_excercise_type == 'american':
                    data_array[i][j] = max(self.option_payoff(asset_prices[i][j]) ,
                                           self.compute_option_price(p, price_up, price_down))
                else:
                    raise Exception('option_excercise_type must be european or american.')
        return data_array

    def create_asset_lite_tree(self):
        """
        Faster method to compute asset prices avoiding doing repeated calculations. If the option is european, it returns just the asset price at maturity date.

        TODO : Find a way to visualize the tree

        Returns 
        ----------------
        [float] :
            [price of the asset at maturity date]
        """
        data_array = []
        

        for i in range(1,self.n+2):
            data_array.append(np.zeros(i))
        data_array[0] = [self.s,] #Initialize the first Node with the actual price

        for i in range(1,len(data_array)):
            for j in range(0,len(data_array[i])):
                if   j==0:
                    data_array[i][j]=data_array[i-1][0]*self.price_change(True)
                elif j==1:
                    data_array[i][j]=data_array[i-1][0]*self.price_change(False)
                else:
                    data_array[i][j]=data_array[i-1][j-1]*self.price_change(False)
        if self.option_excercise_type == 'european':
            return data_array[-1]
        else:
            return data_array

    def create_option_lite_tree(self, asset_prices: list = None):
        """
        Faster method to compute option prices avoiding doing repeated calculations. If the option is european, it returns just the option price at the valuation date.

        TODO : Find a way to visualize the tree
        
        Inputs
        ----------------
        asset_prices : [list[float]] (optional)
            [Asset prices list. When the option is american, values should be passed.]

        Returns
        ----------------
        [float] :
            [price of the option at valuation date]
        """ 
        data_array= []

        #Leaves of the asset prices tree (Spot prices at the last time period)
        if self.option_excercise_type == 'european':
            prices = self.create_asset_lite_tree()
        else:
            prices =self.create_asset_lite_tree()[-1]

        for i in range(1, self.n+2):
            data_array.append(np.zeros(i))

        #initialize the option price at maturity time
        for i in range(len(data_array[-1])): 
            data_array[-1][i] = self.option_payoff(prices[i])

        p = self.risk_neutral_prob() 

        for i in reversed(range(0, len(data_array)-1)):
            for j in range(0, len(data_array[i])):

                if j == 0:
                    price_up = data_array[i+1][0]
                    price_down = data_array[i+1][1]
                else:
                    price_up = price_down
                    price_down = data_array[i+1][j+1]

                if self.option_excercise_type == 'european':
                    data_array[i][j] = self.compute_option_price(p,price_up,price_down)
                else:
                    data_array[i][j] = max(self.option_payoff(asset_prices[i][j]), 
                                            self.compute_option_price(p,price_up,price_down))

        price = data_array[0][0]
        return price

    def return_tree(self, plot = True, text_trees = True):
        """
        Method to compute the asset and option prices at maturity/valuation day (when mode is lite) or the full trees (when mode is full). When mode is full, it creates text files with a representation of the trees (useful for a quick look at small trees). Also contains a method for plotting them.


        Note about plotting and text files: It's recommended to plot small trees. The tree node size and font size can be adjusted. 

        Inputs
        ----------------
        plot : [bool] (optional) 
            [When True, activate the plotting of the trees.]
        text_trees : [bool] (optional)
            [When True, it creates text files with a representation of the trees.]
        
        """

        # Check entries
        self.check_entries(self.mode, self.option_excercise_type, self.option_type, self.positon, self.n, 
                           self.volatility, self.r, self.delta, self.k, self.s)
        
        if self.mode == 'lite':
            self.asset_price = self.create_asset_lite_tree()
            self.option_price = self.create_option_lite_tree(self.asset_price)

        elif self.mode == 'full':
            self.asset_price = self.create_asset_full_tree()                       # Asset prices full tree
            self.option_price = self.create_option_full_tree(self.asset_price)     # Option prices full tree
            
            # Text visualization of the trees
            if text_trees:
                #Flatten arrays
                asset_prices_flat_values = np.round([item for sublist in self.asset_price for item in sublist],3)
                options_prices_flat_values = np.round([item for sublist in self.option_price for item in sublist],3)

                #Create text files with the trees in the array-like format. Useful for small trsettees.
                with open('Asset_prices_tree.txt', 'w') as f:
                    root = build(asset_prices_flat_values)
                    print(root, file=f)
                with open('Options_tree.txt', 'w') as f:
                    root = build(options_prices_flat_values)
                    print(root, file=f)

            # Plot of the trees
            if plot:
                
                #Flatten arrays
                asset_prices_flat_values = np.round([item for sublist in self.asset_price for item in sublist],3)
                options_prices_flat_values = np.round([item for sublist in self.option_price for item in sublist],3)

                BinomialOptionTree.plot_binary_tree(asset_prices_flat_values)
                BinomialOptionTree.plot_binary_tree(options_prices_flat_values)
            





    @staticmethod
    def plot_binary_tree(data: list , node_size = 500, font_size = 5):

        G = nx.DiGraph()
        
        # Create a dictionary to store mapping from unique nodes to values
        unique_to_value = {}

        for val in data:
            unique_node = f"Node_{len(unique_to_value)}"
            if val not in unique_to_value:
                unique_to_value[unique_node] = val
            G.add_node(unique_node)

        # Add edges based on the specific relationships
        for i in range(1, len(data)):
            parent_index = (i - 1) // 2
            G.add_edge(list(G.nodes())[parent_index], list(G.nodes())[i])

        pos = BinomialOptionTree.hierarchy_pos(G, list(G.nodes())[0])

        labels = {node: str(unique_to_value[node]) for node in G.nodes()}
        
        nx.draw(G, pos, with_labels=True, labels=labels, node_size=node_size, node_color='lightblue', font_size=font_size, font_color='black', font_weight='bold')
        plt.autoscale()
        plt.show()
        plt.clf() 
    
    @staticmethod
    def _hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5, 
                       pos=None, parent=None, parsed=[]):
        if pos is None:
            pos = {root: (xcenter, vert_loc)}
        else:
            pos[root] = (xcenter, vert_loc)
        children = list(G.neighbors(root))
        if len(children) != 0:
            dx = width / 2
            nextx = xcenter - width/2 - dx/2
            for child in children:
                nextx += dx
                pos = BinomialOptionTree._hierarchy_pos(G, child, width=dx, vert_gap=vert_gap, vert_loc=vert_loc-vert_gap, xcenter=nextx, pos=pos, parent=root, parsed=parsed)
        return pos

    @staticmethod
    def hierarchy_pos(G, root, width=1., vert_gap=0.2, vert_loc=0, xcenter=0.5):
        pos = BinomialOptionTree._hierarchy_pos(G, root, width, vert_gap, vert_loc, xcenter)
        return pos

    
