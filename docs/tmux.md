
# Tmux Guide

## Installation

### macOS
```bash
brew install tmux
```

### Ubuntu/Debian
```bash
sudo apt-get install tmux
```

### From Source
```bash
git clone https://github.com/tmux/tmux.git
cd tmux
sh autogen.sh
./configure && make
sudo make install
```

## Basic Usage

### Start a session
```bash
tmux new-session -s session-name
```

### List sessions
```bash
tmux list-sessions
```

### Attach to a session
```bash
tmux attach-session -t session-name
```

### Detach from a session
Press `Ctrl + B`, then `D`

## Key Bindings

| Command | Action |
|---------|--------|
| `Ctrl + B` + `C` | Create new window |
| `Ctrl + B` + `N` | Next window |
| `Ctrl + B` + `P` | Previous window |
| `Ctrl + B` + `%` | Split vertically |
| `Ctrl + B` + `"` | Split horizontally |
| `Ctrl + B` + `Arrow` | Navigate panes |
