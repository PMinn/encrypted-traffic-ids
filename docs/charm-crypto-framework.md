# charm-crypto-framework

1. Ensure you have the latest pip, setuptools, and wheel
```sh
pip install --upgrade pip setuptools wheel
```

2. Install System Dependencies (Ubuntu)
```sh
sudo apt-get update
sudo apt-get install build-essential flex bison wget m4 python3 python3-dev libgmp-dev libssl-dev
```

3. Install the Stanford PBC Library
```sh
wget https://crypto.stanford.edu/pbc/files/pbc-1.0.0.tar.gz
tar xzf pbc-1.0.0.tar.gz
cd pbc-1.0.0
./configure LDFLAGS="-lgmp"
make
sudo make install
sudo ldconfig # (or just 'sudo make install' on macOS)
cd ..
```

4. nstall charm-crypto-framework
```sh
pip install charm-crypto-framework
```