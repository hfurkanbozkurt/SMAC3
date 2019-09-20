from hpolib.benchmarks.synthetic_functions.bohachevsky import Bohachevsky
from hpolib.benchmarks.synthetic_functions.branin import Branin
from hpolib.benchmarks.synthetic_functions.camelback import Camelback
from hpolib.benchmarks.synthetic_functions.forrester import Forrester
from hpolib.benchmarks.synthetic_functions.goldstein_price import GoldsteinPrice
from hpolib.benchmarks.synthetic_functions.hartmann3 import Hartmann3
from hpolib.benchmarks.synthetic_functions.hartmann6 import Hartmann6
from hpolib.benchmarks.synthetic_functions.levy import Levy
from hpolib.benchmarks.synthetic_functions.rosenbrock import Rosenbrock
from hpolib.benchmarks.synthetic_functions.sin_one import SinOne
from hpolib.benchmarks.synthetic_functions.sin_two import SinTwo


synthetic_benchmarks = {
    "Branin": Branin,
    "Hartmann3": Hartmann3,
    "Hartmann6": Hartmann6,
    "Camelback": Camelback,
    "Levy": Levy,
    "Bohachevsky": Bohachevsky,
    "SinOne": SinOne,
    "SinTwo": SinTwo,
    "GoldsteinPrice": GoldsteinPrice,
    "Rosenbrock": Rosenbrock,
    "Forrester": Forrester,
}

for k, v in synthetic_benchmarks.copy().items():
    synthetic_benchmarks[k.lower()] = v
