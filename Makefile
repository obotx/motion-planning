BOOTSTRAP_PYTHON ?= python3.10
PYTHON ?= .venv/bin/python
PYTHONPATH_VALUE ?= src

.PHONY: check-python setup smoke gui m1 morph-i morph-ii traj-i traj-ii record-i record-ii record-traj-i record-traj-ii

check-python:
	@command -v $(BOOTSTRAP_PYTHON) >/dev/null 2>&1 || { \
		echo "ERROR: $(BOOTSTRAP_PYTHON) was not found."; \
		echo "Install Python 3.10, or run: make setup BOOTSTRAP_PYTHON=/path/to/python3.10"; \
		echo "Python 3.10 is recommended because OMPL wheels are Python-version specific."; \
		exit 1; \
	}

setup: check-python
	$(BOOTSTRAP_PYTHON) -m venv .venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt

smoke:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) tools/smoke_test.py

gui:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/gui/play.py

m1:
	OMPL_BRIDGE_MODE=native PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/gui/play_m1.py

morph-i:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/simulations/morph_i_free_move.py --run glfw

morph-ii:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/simulations/morph_ii_free_move.py --run glfw

traj-i:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/simulations/morph_i_market_trajectory.py --run glfw --control trajectory

traj-ii:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/simulations/morph_ii_kitchen_trajectory.py --run glfw --control trajectory

record-i:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/simulations/morph_i_free_move.py --run cv --record

record-ii:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/simulations/morph_ii_free_move.py --run cv --record

record-traj-i:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/simulations/morph_i_market_trajectory.py --run cv --control trajectory --record

record-traj-ii:
	PYTHONPATH=$(PYTHONPATH_VALUE) $(PYTHON) src/simulations/morph_ii_kitchen_trajectory.py --run cv --control trajectory --record
