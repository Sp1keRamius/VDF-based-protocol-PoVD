from .consensus_abc import Consensus
from .VDF import VDF
from .pow import POW
# from .virtualpow import VirtualPoW
# try:
#     from .random_oracle import RandomOracleRoot, RandomOracleMining, RandomOracleVerifying
#     from .solidpow import SolidPoW
# except ImportError:
#     print('''Warning: fail to import module random_oracle. SolidPoW is not avaiable.''')