from .objectives import *
import os
# This check the amount of physical RAM installed, as somehow the process crashes if the system memory is small.
mem_gigabytes = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024. ** 3)

if mem_gigabytes > 32:
    from .nas201_new import NAS201, NAS201edge
else:
    from .nas201 import NAS201, NAS201edge

from .nas101 import NAS101Cifar10
