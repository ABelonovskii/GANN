from Learning import Learning
from Parameters import Parameters
import yaml

if __name__ == '__main__':

    learning = Learning()

    if Parameters.regime == 1:
        learning.evolution()
    elif Parameters.regime == 0:
        learning.use()

