"""Entry point: train + validate the parameterized microchannel PINN.

Run inside the Linux VM (OrbStack machine `dev`), from the repo root:
    /root/venv-rbc/bin/python -m pinn.run_microchannel
or:
    /root/venv-rbc/bin/python pinn/run_microchannel.py
"""
from pinn.microchannel import main

if __name__ == "__main__":
    main()
