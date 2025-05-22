# jamming-detection-xapp

Here are the steps to set up the network 5G network using srsRAN Project (gNB), srsRAN 4G (UE), Open5GS (Core), and FlexRIC (Near-RT RIC) to run the xApp.

## Overview

The setup involves:
1.  **Open5GS**: 5G Core Network (running in Docker).
2.  **srsRAN Project gNB**: 5G gNodeB with E2 agent capabilities.
3.  **srsRAN UE (from srsRAN 4G)**: User Equipment.
4.  **FlexRIC**: O-RAN compliant Near-Real-Time Radio Intelligent Controller.
6.  **ZeroMQ**: Used for RF simulation between gNB and UE.
7.  **iperf3**: To generate UE traffic.

## Prerequisites

1.  **Ubuntu 22.04 LTS** (or similar Linux distribution).
2.  **Build tools**: `cmake`, `make`, `gcc` (FlexRIC will require `gcc-10`), `g++`, `swig`, `libsctp-dev`, `python3-dev`, `pkg-config`, `libconfig-dev`, `libconfig++-dev`, `autoconf`, `libtool`.
3.  **srsRAN Project** (latest version with E2 support).
4.  **srsRAN 4G** (latest version for srsUE).
5.  **FlexRIC** (specifically `br-flexric` branch, commit `1a3903a7` or similar).
6.  **Open5GS** (Docker setup recommended).
7.  **Docker** and **Docker Compose**.
8.  **iperf3**.

### Installing ZeroMQ

ZeroMQ is crucial for RF simulation between the gNB and UE in this setup. here is how to install it:

**Install ZeroMQ development libraries via apt**
```bash
sudo apt-get update
sudo apt-get install libzmq3-dev
```

**Build ZeroMQ and CppZMQ from source**

*   **Install libzmq (core ZeroMQ library):**
    ```bash
    sudo apt-get install -y autoconf libtool # Ensure build tools for libzmq are present
    git clone https://github.com/zeromq/libzmq.git
    cd libzmq
    ./autogen.sh
    ./configure
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    cd .. # Go back to the parent directory
    ```

*   **Install czmq (C bindings - often a dependency for srsRAN if building ZMQ from source):**
    ```bash
    git clone https://github.com/zeromq/czmq.git
    cd czmq
    ./autogen.sh
    ./configure
    make -j$(nproc)
    sudo make install
    sudo ldconfig
    cd .. # Go back to the parent directory
    ```

**Important:** After installing ZeroMQ, if you had previously attempted to build srsRAN Project or srsRAN 4G, you **must re-run `cmake` and `make`** for both projects to ensure they detect and link against the ZeroMQ libraries correctly.

## Setup Steps

### 1. Build srsRAN Project (gNB)

Ensure ZeroMQ is installed (see "Installing ZeroMQ" above) before building.
```bash
git clone https://github.com/srsran/srsRAN_Project.git
cd srsRAN_Project
mkdir build
cd build
# Ensure ZMQ is found by CMake. If built from source, it should be in /usr/local.
# The -DENABLE_ZEROMQ=ON flag is key.
cmake ../ -DENABLE_ZEROMQ=ON # Add other srsRAN Project build flags as needed
make -j$(nproc)
# sudo make install (optional, if you prefer)
cd ../.. # Go back to the directory where you cloned srsRAN_Project
```
*During the `cmake` step for srsRAN_Project, look for output confirming ZeroMQ was found, similar to:*
```
-- FINDING ZEROMQ.
-- Found libZEROMQ: /usr/local/include, /usr/local/lib/libzmq.so (or /usr/lib/x86_64-linux-gnu/libzmq.so if from apt)
```

### 2. Build srsRAN 4G (for srsUE)

Ensure ZeroMQ is installed (see "Installing ZeroMQ" above) before building.
```bash
git clone https://github.com/srsran/srsran.git # This is srsRAN 4G
cd srsran
mkdir build
cd build
# CMake should automatically find ZMQ if installed system-wide
cmake ../ # Add srsRAN 4G build flags as needed
make -j$(nproc)
# sudo make install (optional, if you prefer)
cd ../.. # Go back to the directory where you cloned srsran
```
*Similarly, check the `cmake` output for srsRAN 4G to confirm ZeroMQ detection.*

### 3. Build FlexRIC
```bash
# Ensure FlexRIC dependencies are installed (some may overlap with ZMQ build tools)
sudo apt-get update
sudo apt-get install swig libsctp-dev python3 cmake-curses-gui python3-dev pkg-config libconfig-dev libconfig++-dev autoconf libtool

# Switch to gcc-10 because build will fail with gcc-11/12, next two commented lines demostrate how: 
# sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-10 10
# sudo update-alternatives --config gcc

git clone https://gitlab.eurecom.fr/mosaic5g/flexric.git
cd flexric
git checkout br-flexric # Or the commit specified in srsRAN docs (e.g., 1a3903a7)
mkdir build
cd build
cmake -DKPM_VERSION=KPM_V3_00 -DXAPP_DB=NONE_XAPP ../
make -j$(nproc)
sudo make install
cd ../.. # Go back to the directory where you cloned flexric
```
### 4. Configuration Files

Create the following configuration files in their respective locations. Adjust paths as necessary. 

#### a. gNB Configuration (`~/srsRAN_Project/build/apps/gnb/gnb_zmq.yaml`)

This is a minimal example. You'll need to adapt it from a full srsRAN Project gNB ZMQ example. The key is the `e2` section.
```yaml name=gnb_zmq.yaml
# ... (other gNB parameters like rf_driver, cell_cfg, ru_cfg, core_conn, etc.)
# Example:
# amf_addr: 10.53.1.2 # IP of AMF from Open5GS Docker network
# gnb_name: "srsRAN-gnb"
# cell_cfg:
#   dl_arfcn: 368500
#   band: 3
#   channel_bandwidth_MHz: 20
#   pci: 1
#   # ... other cell parameters

# E2 Agent Configuration
e2:
  enable_du_e2: true                # Enable DU E2 agent
  e2sm_kpm_enabled: true            # Enable KPM service module
  e2sm_rc_enabled: true             # Enable RC service module
  addr: "127.0.0.1"                 # RIC IP address
  port: 36421                       # RIC E2 port
  # bind_addr: "127.0.0.1"          # Optional: gNB's IP for E2 binding

# Optional: Enable E2 PCAP for debugging
# pcap:
#   e2ap_enable: true
#   e2ap_du_filename: /tmp/gnb_du_e2ap.pcap

# Optional: Enable RLC metrics for KPM (if not enabled by default)
# metrics:
#   rlc_report_period: 1000
```

#### b. UE Configuration (`~/srsRAN_4G/build/srsue/ue_zmq.conf`)

Adapt from a standard srsUE ZMQ configuration file. Ensure `usim` parameters match Open5GS.
```ini name=ue_zmq.conf
# ... (standard srsUE ZMQ config)
# Example:
# [rf]
# device_name = zmq
# device_args = tx_port=tcp://127.0.0.1:2001,rx_port=tcp://127.0.0.1:2000,base_srate=23.04e6

# [usim]
# algo = milenage
# op = "63bfa50ee6523365ff14c1f45f88737d" # Example OP, match Open5GS
# k = "00112233445566778899aabbccddeeff"  # Example K, match Open5GS
# imsi = 001010123456780                   # Example IMSI, match Open5GS
# mode = 5g
# apn = internet # Match Open5GS APN

# [log]
# filename = /tmp/ue.log
# all_level = info
```

## Running everything (order of operations)

Open multiple terminals for each component. Adjust paths to your build directories.

1.  **Terminal 1: Start Open5GS (5G Core)**
    ```bash
    cd ~/srsRAN_Project/docker # Or your srsRAN Project docker directory
    docker compose up 5gc
    ```
    Wait for it to initialize (you should see "Connection to 127.0.0.1 27017 port [tcp/*] succeeded!").

2.  **Terminal 2: Start FlexRIC (Near-RT RIC)**
    ```bash
    cd ~/flexric/build/examples/ric/ # Or your FlexRIC build path
    sudo ./nearRT-RIC
    ```
    Wait for it to initialize (you should see it loading SMs and listening).

3.  **Terminal 3: Start srsRAN gNB**
    ```bash
    cd ~/srsRAN_Project/build/apps/gnb/ # Or your srsRAN Project build path
    sudo ./gnb -c gnb_zmq.yaml e2 --addr="127.0.0.1" --bind_addr="127.0.0.1"
    ```
    Watch for "N2: Connection to AMF..." and "E2AP: Connection to Near-RT-RIC..." messages. The RIC terminal should also show an E2 Setup Request.

4.  **Terminal 4: Start srsRAN UE**
    Create a network namespace for the UE first (if not already done):
    ```bash
    sudo ip netns add ue1
    ```
    Then start the UE:
    ```bash
    cd ~/srsRAN_4G/build/srsue/src/ # Or your srsRAN 4G build path
    sudo ./srsue ../ue_zmq.conf # Assuming ue_zmq.conf is in ../
    ```
    Watch for "RRC Connected" and "PDU Session Establishment successful. IP: 10.45.x.x".

5.  **Terminal 5: Start iperf Traffic (from UE to a reachable IP)**
    The IP `10.45.1.1` is often the `ogstun` interface IP on the Open5GS Docker container.
    ```bash
    sudo ip netns exec ue1 iperf3 -c 10.45.1.1 -t 3600 -i 5 -u -b 10M # UDP, 10 Mbps
    # Or TCP:
    # sudo ip netns exec ue1 iperf3 -c 10.45.1.1 -t 3600 -i 5
    ```

6.  **Terminal 6: Start FlexRIC xApp**

    Navigate to the xApp directory:
    ```bash
    cd ~/flexric/build/examples/xApp/c/monitor/
    ```

    *   Run KPM xApp directly:
        ```bash
        ./xapp_oran_moni -c ./xapp_mon_e2sm_kpm.conf
        ```
        This will run once, print KPM metrics (throughput) if received, and then stop.


## Troubleshooting & Key Considerations

*   **Firewall**: Ensure no firewalls are blocking SCTP traffic on port 36421 (E2) or TCP on 36422 (E42) between gNB and RIC, or ZMQ ports (2000, 2001) if gNB/UE are on different machines.
*   **IP Addresses**: Double-check all IP addresses in configuration files (gNB's AMF address, RIC address, E2 bind address if necessary).
*   **srsRAN Documentation**: The srsRAN Project "O-RAN NearRT-RIC and xApp" tutorial is your primary reference for their specific E2 implementation details and limitations.
*   **UE Activity**: Radio metrics (especially RSRP/RSRQ/SINR) are typically reported when a UE is active and generating traffic. Ensure iperf or other traffic is running.
*   **FlexRIC Version**: Ensure you are using the FlexRIC version/branch compatible with the srsRAN Project's E2 agent.
*   **gNB Logs**: The srsRAN gNB console output is very verbose and can provide clues if E2 messages are being exchanged or if there are errors in processing subscriptions.
