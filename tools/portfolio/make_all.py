"""Build legacy figure candidates and animations."""
import legacy_figures
import animations
import re_sweep
import comsol_2023
import scaling
import gci

if __name__ == '__main__':
    animations.main()
    re_sweep.main()
    legacy_figures.main()
    comsol_2023.main()
    scaling.main()
    gci.main()
