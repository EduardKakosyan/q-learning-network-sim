class CustomRNG:
    """
    A custom pseudo-random number generator (PRNG) class that uses a Linear Congruential Generator (LCG) algorithm.
    This class supports generating random floats between 0 and 1 and selecting random items from a list.
    """

    def __init__(self, seed: int):
        """
        Initialize the PRNG with a seed value.
        
        Args:
            seed (int): The initial seed value for the generator.
        """
        self.state = seed

    def random(self) -> float:
        """
        Generate a pseudo-random float between 0 and 1 using the LCG algorithm.
        
        Returns:
            float: A pseudo-random number in the range [0, 1).
        """
        # Linear Congruential Generator (LCG) algorithm parameters
        a = 1664525  # Multiplier
        c = 1013904223  # Increment
        m = 2**32  # Modulus
        self.state = (a * self.state + c) % m
        return self.state / m

    def choice(self, items: list):
        """
        Select a random item from a non-empty list.
        
        Args:
            items (list): The list of items to choose from.
        
        Returns:
            Any: A randomly selected item from the list.
        
        Raises:
            ValueError: If the input list is empty.
        """
        if not items:
            raise ValueError("Cannot choose from an empty list")
        index = int(self.random() * len(items))
        return items[index]
