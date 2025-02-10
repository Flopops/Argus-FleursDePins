#!/bin/bash
# Exit 1 : installation failed, no python or venv lib
# Exit 2 : installation failed, venv not created


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
VENV_DIR="${SCRIPT_DIR}/argus_venv/"
VENV_ACTIVATE="${VENV_DIR}/bin/activate"

check_python_and_venv() {
  echo "Checking Python and venv..."
  if ! command -v python3 &> /dev/null || ! python3 -c "import venv" &>/dev/null; then
    printf "\e[1m\e[31mERROR: Python 3 or venv module not found. Aborting...\e[0m\n"
    exit 1
  fi
}

create_and_activate_venv() {
  echo "Environment not install yet. Creating..."
  python3 -m venv "$VENV_DIR"
  if [[ -f "$VENV_ACTIVATE" ]]; then
    echo "Activating just created virtual environment..."
    source "$VENV_ACTIVATE"
    echo "Virtual environment created at $VENV_DIR"
  else
    echo "Failed to create virtual environment. Exiting..."
    exit 2
  fi
}

install_dependencies() {
  python3 -m pip install -r "${SCRIPT_DIR}/requirements_lin.txt"
  python3 -m pip uninstall opencv-python -y # To prevent "xcb" error
}

main() {
  

  check_python_and_venv
  if [[ -f "$VENV_ACTIVATE" ]]; then
    echo "Activating existing virtual environment..."
    source "$VENV_ACTIVATE"
  else
    create_and_activate_venv
    install_dependencies
  fi

  # TODO: use a launch script
  echo "Launching application..."
  python3 "${SCRIPT_DIR}/src/app.py"
}

main
