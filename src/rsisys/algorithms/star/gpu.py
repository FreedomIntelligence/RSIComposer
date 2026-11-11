"""Check that the CUDA ordinal resolves to the scheduler-allocated PCI device."""
import csv
import ctypes
import os
import re
import subprocess


def pci_address(raw):
    match = re.fullmatch(r'([0-9a-f]+):([0-9a-f]{2}):([0-9a-f]{2})\.([0-7])',
                         raw.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f'Invalid GPU PCI address: {raw!r}')
    domain, bus, slot, function = match.groups()
    return f'{int(domain, 16):04x}:{bus.lower()}:{slot.lower()}.{function}'


def cuda_pci_devices():
    library = ctypes.CDLL('libcuda.so.1')

    def call(name, types, *arguments):
        function = getattr(library, name)
        function.argtypes = types
        function.restype = ctypes.c_int
        code = function(*arguments)
        if code:
            raise RuntimeError(f'{name} returned CUDA error {code}')

    integer = ctypes.c_int
    pointer = ctypes.POINTER(integer)
    call('cuInit', (ctypes.c_uint,), 0)
    count = integer()
    call('cuDeviceGetCount', (pointer,), ctypes.byref(count))
    addresses = []
    for index in range(count.value):
        device = integer()
        call('cuDeviceGet', (pointer, integer), ctypes.byref(device), index)
        bus = ctypes.create_string_buffer(32)
        call('cuDeviceGetPCIBusId', (ctypes.c_void_p, integer, integer), bus, len(bus), device)
        addresses.append(pci_address(bus.value.decode()))
    return addresses


def allocation_plan(visible, nvml_rows):
    allocated = visible.split(',')
    uuid_pattern = r'GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}'
    if (len(allocated) != 4 or len(set(ident.lower() for ident in allocated)) != 4
            or any(not re.fullmatch(uuid_pattern, ident, re.IGNORECASE) for ident in allocated)):
        raise PermissionError('Expected exactly four distinct scheduler-allocated GPU UUIDs')
    inventory = {}
    for index, uuid, bus in csv.reader(nvml_rows.splitlines()):
        uuid, index = uuid.strip(), index.strip()
        if uuid.lower() in inventory or not index.isdecimal():
            raise RuntimeError('Invalid NVML GPU inventory')
        inventory[uuid.lower()] = {'uuid': uuid, 'index': index, 'pci_bus_id': pci_address(bus)}
    if any(ident.lower() not in inventory for ident in allocated):
        raise PermissionError('An allocated GPU is absent from NVML')
    result = [inventory[ident.lower()] for ident in allocated]
    if len({row['index'] for row in result}) != 4 or len({row['pci_bus_id'] for row in result}) != 4:
        raise PermissionError('Allocated GPU ordinals or PCI addresses overlap')
    return result


def admit_gpu():
    inventory = subprocess.check_output(['nvidia-smi', '--query-gpu=index,uuid,pci.bus_id',
                                         '--format=csv,noheader'], text=True, timeout=15)
    devices = allocation_plan(os.getenv('NVIDIA_VISIBLE_DEVICES', ''), inventory)
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(device['index'] for device in devices)
    if cuda_pci_devices() != [device['pci_bus_id'] for device in devices]:
        raise PermissionError('Visible CUDA ordinals differ from allocated GPU PCI addresses')
    return devices

