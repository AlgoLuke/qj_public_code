#!/bin/bash

# BEFORE RUNNING: Edit the configuration variables in the [CONFIG] section below.
# This script is designed to be run on a fresh Ubuntu server. Do NOT skip this step.

# ===================================================================================
#
#          FILE: setup_server.sh
#
#         USAGE: sudo ./setup_server.sh
#
#   DESCRIPTION: An all-in-one script to automate the setup and hardening of a
#                fresh Ubuntu server. It configures security, Docker, users,
#                and deploys a WireGuard VPN server for secure access.
#
#       CREATED: 2025-06-27
#      REVISION: 3.2 (Systemd compatibility fallback)
#
# ===================================================================================

# -----------------------------------------------------------------------------------
# | [CONFIG] PLEASE EDIT THESE VARIABLES BEFORE RUNNING THE SCRIPT                  |
# -----------------------------------------------------------------------------------

# --- SSH & User Configuration ---
NEW_SSH_PORT=2222
SSH_PUBLIC_KEY="ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIKdBjhydBrtYCMG4VZmwo9dzJjyEOH2W2YMsILsJB/Iq jakub@quantjourney.pro"
DISABLE_PASSWORD_AUTH=true
USER_ONE="jakub"

# --- VPN Configuration ---
VPN_PORT=51820
FIRST_VPN_CLIENT_NAME="macbook"

# --- Optional Components ---
INSTALL_MONITORING_STACK=true

# -----------------------------------------------------------------------------------
# | [SCRIPT LOGIC]                                                                  |
# -----------------------------------------------------------------------------------

# --- Color Definitions ---
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
WHITE='\033[1;37m'
NC='\033[0m' # No Color

# --- Logging ---
LOG_FILE="/var/log/server-setup.log"
exec > >(tee -a "$LOG_FILE")
exec 2>&1
echo -e "${CYAN}Starting setup at $(date)${NC}"

# --- Detect whether systemd is available ---
has_systemd=false
if pidof systemd >/dev/null 2>&1 && [ -d /run/systemd/system ]; then
    has_systemd=true
fi

# --- Script Checks ---
if [ -z "$BASH_VERSION" ]; then
  echo -e "${RED}Please run this script with bash, not sh.${NC}"
  exit 1
fi

if [ "$EUID" -ne 0 ]; then
  echo -e "${RED}ERROR: Please run this script as root using sudo.${NC}"
  exit 1
fi

if ! grep -q "Ubuntu" /etc/os-release; then
    echo -e "${RED}ERROR: This script is designed for Ubuntu only.${NC}"
    exit 1
fi

UBUNTU_VERSION=$(lsb_release -rs)

AVAILABLE_SPACE=$(df / | awk 'NR==2 {print $4}')
if [ $AVAILABLE_SPACE -lt 5242880 ]; then
    echo -e "${RED}ERROR: Insufficient disk space. At least 5GB required.${NC}"
    exit 1
fi

if ! curl -s --head https://google.com | grep "HTTP/" > /dev/null; then
    echo -e "${RED}ERROR: No internet connection detected.${NC}"
    exit 1
fi

cleanup() {
    echo -e "${RED}Script failed at line $1"
    echo -e "Please check the logs and fix any issues before re-running${NC}"
    exit 1
}
trap 'cleanup $LINENO' ERR

validate_ssh_key() {
    if [[ ! "$SSH_PUBLIC_KEY" =~ ^ssh-(rsa|ed25519|ecdsa) ]]; then
        echo -e "${RED}ERROR: Invalid SSH public key format.${NC}"
        exit 1
    fi
}

set -e

log() {
    echo -e "${BLUE}--------------------------------------------------"
    echo -e "${WHITE}$1${NC}"
    echo -e "${BLUE}--------------------------------------------------${NC}"
}

success() {
    echo -e "${GREEN}✓ $1${NC}"
}

warn() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

error() {
    echo -e "${RED}✗ $1${NC}"
}

log "Starting server setup, hardening, and VPN installation..."

sudo apt-get update
for cmd in curl wget openssl htpasswd; do
    if ! command -v $cmd &> /dev/null; then
        error "ERROR: $cmd is required but not installed"
        exit 1
    fi
done
sudo apt-get install -y curl

log "[1/10] Updating system and installing essential utilities..."
sudo apt-get dist-upgrade -y
sudo apt-get install -y htop ncdu git unzip unattended-upgrades
success "System updated and utilities installed."


log "[2/10] Disabling and masking snapd..."
if command -v snap >/dev/null; then
    if [ "$has_systemd" = true ]; then
        sudo systemctl stop snapd.service
        sudo systemctl disable snapd.service
        sudo systemctl mask snapd.service
    else
        sudo service snapd stop || true
    fi
    sudo apt-get autoremove --purge -y snapd
    success "snapd has been disabled and purged."
else
    warn "snapd not found, skipping removal."
fi

log "[3/10] Applying security hardening measures..."
sudo sed -i 's/^#\?PermitRootLogin.*/PermitRootLogin no/' /etc/ssh/sshd_config
sudo sed -i "s/^#\?Port 22/Port $NEW_SSH_PORT/" /etc/ssh/sshd_config
if [ "$DISABLE_PASSWORD_AUTH" = true ]; then
    if [ -z "$SSH_PUBLIC_KEY" ]; then
        error "SSH_PUBLIC_KEY not set"
        exit 1
    fi
    sudo sed -i 's/^#\?PasswordAuthentication .*/PasswordAuthentication no/' /etc/ssh/sshd_config
fi
validate_ssh_key
sudo sed -i 's/^#\?MaxAuthTries.*/MaxAuthTries 3/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?ClientAliveInterval.*/ClientAliveInterval 300/' /etc/ssh/sshd_config
sudo sed -i 's/^#\?ClientAliveCountMax.*/ClientAliveCountMax 2/' /etc/ssh/sshd_config
success "SSH server hardened."

if [ "$has_systemd" = true ]; then
    sudo dpkg-reconfigure --priority=low unattended-upgrades
else
    warn "Skipping unattended-upgrades reconfiguration (no systemd)"
fi
success "Automatic security updates configured."

sudo apt-get install -y fail2ban
if [ "$has_systemd" = true ]; then
    sudo systemctl enable fail2ban
    sudo systemctl start fail2ban
else
    sudo service fail2ban start || true
fi
success "Fail2ban installed and enabled."

cat > /etc/fail2ban/jail.local << EOF
[DEFAULT]
bantime = 3600
maxretry = 3
findtime = 600

[sshd]
enabled = true
port = $NEW_SSH_PORT
EOF

if [ "$has_systemd" = true ]; then
    sudo systemctl restart fail2ban
else
    sudo service fail2ban restart || true
fi

sudo tee /etc/sysctl.d/99-hardening.conf > /dev/null <<EOF
net.ipv4.conf.default.rp_filter=1
net.ipv4.conf.all.rp_filter=1
net.ipv4.icmp_echo_ignore_broadcasts=1
net.ipv4.conf.all.accept_source_route=0
net.ipv4.conf.all.send_redirects=0
net.ipv4.tcp_syncookies=1
EOF
sudo sysctl --system
success "Kernel parameters hardened."

log "[4/10]Configuring log rotation and management..."
# Configure logrotate for application logs
sudo tee /etc/logrotate.d/quant-apps << EOF
/var/log/quant/*.log {
    daily
    rotate 30
    compress
    delaycompress
    missingok
    notifempty
    create 0644 $USER_ONE $USER_ONE
}
EOF

# Ensure journal logs don't consume excessive disk space
sudo sed -i 's/^#SystemMaxUse=.*/SystemMaxUse=1G/' /etc/systemd/journald.conf
sudo sed -i 's/^#SystemMaxFileSize=.*/SystemMaxFileSize=100M/' /etc/systemd/journald.conf
if [ "$has_systemd" = true ]; then
    sudo systemctl restart systemd-journald
else
    sudo service rsyslog restart || true
fi
success "Log rotation configured."

log "[5/10] Configuring Firewall and restarting SSH..."
sudo ufw --force reset
sudo ufw allow $NEW_SSH_PORT/tcp
sudo ufw allow $VPN_PORT/udp
if [ "$INSTALL_MONITORING_STACK" = true ]; then
    sudo ufw allow 9090/tcp
    sudo ufw allow 3000/tcp
    sudo ufw allow 80/tcp
fi
sudo ufw --force enable

if [ "$has_systemd" = true ]; then
    sudo systemctl restart sshd.service
else
    sudo service ssh restart || true
fi
success "Firewall enabled. SSH restarted."

log "[6/10] Installing Docker..."
if command -v docker &> /dev/null; then
    warn "Docker already installed"
else
    sudo apt-get install -y ca-certificates gnupg
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
    sudo chmod a+r /etc/apt/keyrings/docker.gpg
    echo \
    "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://download.docker.com/linux/ubuntu \
    $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo apt-get update
    sudo apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
    getent group docker || sudo groupadd docker
    success "Docker installed."
fi

log "[7/10] Creating user and setting up SSH and shell..."
if id "$USER_ONE" &>/dev/null; then
    warn "User $USER_ONE exists"
else
    sudo adduser --disabled-password --gecos "" "$USER_ONE"
fi
sudo apt-get install -y zsh
sudo usermod -aG sudo,docker "$USER_ONE"
sudo usermod --shell $(which zsh) "$USER_ONE"

if [ "$DISABLE_PASSWORD_AUTH" = true ]; then
    ssh_dir="/home/$USER_ONE/.ssh"
    auth_keys_file="$ssh_dir/authorized_keys"
    sudo -u "$USER_ONE" mkdir -p "$ssh_dir"
    sudo -u "$USER_ONE" touch "$auth_keys_file"
    echo "$SSH_PUBLIC_KEY" | sudo tee -a "$auth_keys_file" > /dev/null
    sudo chmod 700 "$ssh_dir"
    sudo chmod 600 "$auth_keys_file"
    sudo chown -R "$USER_ONE:$USER_ONE" "$ssh_dir"
fi

if [ ! -d "/home/$USER_ONE/.oh-my-zsh" ]; then
    sudo -u $USER_ONE sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh) --unattended"
fi
success "User and shell setup complete."

log "[8/10] Installing WireGuard VPN via PiVPN..."
PUBLIC_IP=$(curl -4s icanhazip.com)
IPv4dev=$(ip route get 8.8.8.8 | awk '{print $5; exit}')
IPv4addr=$(ip route get 8.8.8.8 | awk '{print $7; exit}')
cat > /tmp/setupVars.conf << EOF
PLAT=Ubuntu
OSCN=$(lsb_release -cs)
USING_UFW=1
IPv4dev=$IPv4dev
IPv4addr=$IPv4addr
IPv4gw=$(ip route get 8.8.8.8 | awk '{print $3; exit}')
pivpnPROTO=udp
pivpnPORT=$VPN_PORT
pivpnDNS1=1.1.1.1
pivpnDNS2=1.0.0.1
pivpnHOST=$PUBLIC_IP
pivpnPERSISTENTKEEPALIVE=25
pivpnENCRYPT=256
UNATTUPG=1
pivpnUSER=$USER_ONE
EOF

curl -L https://install.pivpn.io | bash /dev/stdin --unattended /tmp/setupVars.conf
rm -f /tmp/setupVars.conf
pivpn add -n "$FIRST_VPN_CLIENT_NAME"
success "WireGuard installed and VPN profile created."

log "[9/10] Generating final instructions..."
VPN_CONFIG_PATH="/home/$USER_ONE/configs/$FIRST_VPN_CLIENT_NAME.conf"

if [ "$INSTALL_MONITORING_STACK" = true ]; then
    log "[10/10] Installing monitoring stack..."
    sudo apt-get install -y nginx apache2-utils
    docker run -d --name node-exporter --restart=always \
        --net="host" --pid="host" -v "/:/host:ro,rslave" \
        prom/node-exporter
    docker run -d --name prometheus --restart=always \
        -p 9090:9090 -v prometheus-data:/prometheus prom/prometheus
    docker run -d --name grafana --restart=always \
        -p 3000:3000 -v grafana-data:/var/lib/grafana grafana/grafana
    sleep 10
    MONITORING_USER="admin"
    MONITORING_PASS=$(openssl rand -base64 16)
    htpasswd -bc /etc/nginx/.htpasswd "$MONITORING_USER" "$MONITORING_PASS"
    cat > /root/monitoring-credentials.txt << EOF
Monitoring Credentials:
Username: $MONITORING_USER
Password: $MONITORING_PASS
EOF
    chmod 600 /root/monitoring-credentials.txt
    success "Monitoring stack installed."
fi

cat << DOCSTRING

${GREEN}================================================================================
=== SERVER SETUP COMPLETE =========================================================
================================================================================${NC}

${WHITE}You can now connect securely via SSH and use your VPN profile.${NC}
${CYAN}Public IP: ${YELLOW}$PUBLIC_IP${NC}
${CYAN}VPN Config: ${YELLOW}$VPN_CONFIG_PATH${NC}

DOCSTRING

log "Script finished. Please follow the instructions above."
