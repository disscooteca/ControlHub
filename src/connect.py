import streamlit as st

from bleak import BleakScanner, BleakClient

# Os mesmos UUIDs e Nome usados no código da ESP32
DEVICE_NAME = "ESP32_BLE_Control"
SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHAR_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"

# --- FUNÇÕES ASSÍNCRONAS PARA O BLE ---

async def scan_and_connect():
    """Procura a ESP32 pelo nome e retorna o endereço MAC dela."""
    st.info("Procurando ESP32...")
    devices = await BleakScanner.discover(timeout=3.0)
    for d in devices:
        if d.name == DEVICE_NAME:
            return d
    return None

async def send_command(command: str):
    """Conecta, envia o texto e desconecta."""
    device = await scan_and_connect()
    if device:
        try:
            async with BleakClient(device) as client:
                await client.write_gatt_char(CHAR_UUID, command.encode('utf-8'))
                return True, f"Comando '{command}' enviado com sucesso!"
        except Exception as e:
            return False, f"Erro ao enviar: {e}"
    else:
        return False, "ESP32 não encontrada. Ela está ligada?"

async def read_status():
    """Conecta, lê o texto atual da ESP32 e desconecta."""
    device = await scan_and_connect()
    if device:
        try:
            async with BleakClient(device) as client:
                valor_bytes = await client.read_gatt_char(CHAR_UUID)
                return True, valor_bytes.decode('utf-8')
        except Exception as e:
            return False, f"Erro ao ler: {e}"
    else:
        return False, "ESP32 não encontrada."

# --- INTERFACE DO STREAMLIT ---

