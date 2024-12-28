import sys
from path import Path
sys.path.append(Path(__file__).parent.parent)

from apf.base.bessel import mode

print(mode(4.0, 5.0))
